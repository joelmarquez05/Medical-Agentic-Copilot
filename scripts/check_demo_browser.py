"""Check guided flows/reset in an isolated app with a labelled offline test model."""
import json
from pathlib import Path
import re
import socket
import threading
import time
import uuid
from urllib.request import urlopen

from playwright.sync_api import sync_playwright
import uvicorn

from medical_copilot.web.agent import WorkspaceAgent
from medical_copilot.web.app import create_app
from medical_copilot.web.runtime import public_result

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "tmp/demo-browser-20261016" / uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    calls, errors = [], []
    class TestModel:
        def reply(self, request):
            return {"type": "final", "answer": "TEST-ONLY browser fixture: a previous PROTEOS discussion. This is not a clinical answer.",
                    "justification": "Synthetic transport in an isolated browser test.", "citations": [], "limitations": []}
    def runner(store, job, emit, root):
        calls.append(job["id"])
        session = store.session(job["session_id"])
        with store.memory() as memory:
            record = WorkspaceAgent(TestModel(), memory, session["owner_id"], session["id"],
                patient_selected=bool(session["patient_id"])).run({"question_id": job["id"], "question": job["question"]})
        result = public_result(record, session["id"])
        result["output_origin"] = "synthetic_browser_test_fixture"
        return result
    app = create_app(directory, runner)
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        port = connection.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            try:
                with urlopen(url + "/api/bootstrap", timeout=1):
                    break
            except OSError:
                time.sleep(.05)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 980})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url, wait_until="networkidle")
            assert page.locator(".example-item").count() == 3
            assert page.get_by_role("combobox", name="Patient context").locator("option:checked").inner_text() == "No patient context"

            def choose(title):
                before = page.evaluate("localStorage.getItem('medical-copilot-conversation')")
                page.get_by_role("button", name="Examples", exact=True).click()
                assert page.locator(".drawer-example").count() == 6
                page.locator(".drawer-example").filter(has_text=title).click()
                page.wait_for_function("old => localStorage.getItem('medical-copilot-conversation') !== old", arg=before)
                page.wait_for_function("!document.querySelector('select[aria-label=\"Patient context\"]').disabled")
                return snapshot()

            def snapshot():
                active = page.evaluate("localStorage.getItem('medical-copilot-conversation')")
                return page.request.get(url + "/api/sessions/" + active).json()

            first = choose("Find the PROTEOS study")
            assert first["session"]["patient_id"] is None and first["files"] == [] and calls == []
            second = choose("Read the case and the paper")
            assert len(second["files"]) == 2 and second["facts"]["weight_kg"]["value"] == 64.2
            options = page.get_by_role("combobox", name="Patient context").inner_text()
            assert "(fictional)" not in options
            page.get_by_role("button", name="Attached files", exact=True).click()
            assert page.locator(".document-item").count() == 2
            page.screenshot(path=str(directory / "case-and-paper.png"), full_page=True)
            page.get_by_role("button", name="Close context panel").click()
            page.get_by_role("button", name="Send question", exact=True).click()
            page.wait_for_function("document.querySelector('.exchange .answer-body') && !document.querySelector('.composer.is-busy')")
            assert len(calls) == 1
            third = choose("Continue in a new conversation")
            assert third["session"]["id"] != second["session"]["id"]
            assert third["session"]["patient_id"] == second["session"]["patient_id"]
            assert third["messages"] == [] and len(third["files"]) == 3
            assert third["facts"]["weight_kg"]["value"] == 62.4
            with app.state.store.memory() as memory:
                assert "TEST-ONLY" in json.dumps(memory.context(third["session"]["patient_id"], third["session"]["id"]))
            page.get_by_role("button", name="Appearance & settings", exact=True).click()
            page.get_by_role("button", name="Reset conversations", exact=True).click()
            page.get_by_role("button", name="Cancel", exact=True).click()
            assert len(app.state.store.sessions()) > 0
            page.get_by_role("button", name="Reset conversations", exact=True).click()
            page.get_by_role("button", name="Delete all conversations", exact=True).scroll_into_view_if_needed()
            page.screenshot(path=str(directory / "reset-settings.png"), full_page=True)
            page.get_by_role("button", name="Delete all conversations", exact=True).click()
            page.wait_for_function("!document.querySelector('dialog[open]') && document.querySelectorAll('.session-item').length === 0")
            assert app.state.store.sessions() == [] and len(app.state.store.patients()) == 2
            with app.state.store.memory() as memory:
                assert memory.recent_interactions(third["session"]["patient_id"]) == []
                assert len(memory.documents(third["session"]["patient_id"])) == 3
            page.get_by_role("button", name="Examples", exact=True).click()
            page.screenshot(path=str(directory / "examples-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_function("document.querySelector('.sidebar').getBoundingClientRect().right <= 1")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(directory / "examples-mobile.png"), full_page=True)
            page.get_by_role("button", name="Close context panel").click()
            page.set_viewport_size({"width": 1440, "height": 980})
            restored = choose("Continue a patient conversation")
            assert restored["facts"]["weight_kg"]["value"] == 62.4 and restored["messages"] == []
            with app.state.store.memory() as memory:
                assert memory.recent_interactions(restored["session"]["patient_id"]) == []
            assert not errors and len(calls) == 1
            browser.close()
        receipt = {"date": "2026-10-16", "passed": True, "isolated_workspace": str(directory.relative_to(ROOT)),
                   "original_featured_examples": 3, "new_continuity_steps": 3, "synthetic_test_answers": len(calls),
                   "new_provider_requests": 0, "javascript_errors": errors,
                   "checks": ["No patient context label", "No fictional suffix on patient options", "No inference during setup",
                              "Owned case and study PDFs", "New conversation retains real prior test history",
                              "Dated follow-up without reset", "Reset cancel", "Confirmed reset clears both histories",
                              "Patients and files survive reset", "Examples work again after reset", "Desktop/mobile layout"]}
        (ROOT / "data/checks/demo-browser-20261016.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
    finally:
        server.should_exit = True
        thread.join(8)


if __name__ == "__main__":
    main()
