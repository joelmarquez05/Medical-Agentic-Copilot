"""Browser smoke checks; --live explicitly authorizes five Gemma smoke questions."""
import argparse
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse an already submitted identical smoke question after an interrupted harness; never rerun by answer quality")
    parser.add_argument("--url", default="http://127.0.0.1:8787")
    args = parser.parse_args()
    output = ROOT / "tmp/web-implementation-20261015"
    output.mkdir(exist_ok=True)
    checks, live = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 980})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        page.get_by_role("heading", name="Evidence for your").wait_for()
        original = page.request.get(args.url + "/api/bootstrap").json()["appearance"]

        def active():
            return page.evaluate("localStorage.getItem('medical-copilot-conversation')")

        def snapshot():
            return page.request.get(args.url + "/api/sessions/" + active()).json()

        def new_general():
            previous = active()
            page.locator('.new-consultation').click()
            page.wait_for_function("old => localStorage.getItem('medical-copilot-conversation') !== old", arg=previous)
            page.wait_for_function("!document.querySelector('select[aria-label=\"Patient context\"]').disabled")

        def send(question, expected):
            reused = None
            if args.resume:
                for candidate in page.request.get(args.url + '/api/bootstrap').json()['sessions']:
                    data = page.request.get(args.url + '/api/sessions/' + candidate['id']).json()
                    reused = next((m for m in data['messages'] if m['question'] == question), None)
                    if reused:
                        page.evaluate("id => localStorage.setItem('medical-copilot-conversation', id)", candidate['id'])
                        page.reload(wait_until='networkidle')
                        page.locator('.exchange').first.wait_for()
                        break
            if not reused:
                box = page.get_by_role("textbox", name="Ask a clinical question")
                box.fill(question)
                count = len(snapshot()["messages"]) if active() else 0
                page.get_by_role("button", name="Send question", exact=True).click()
                page.wait_for_function("n => document.querySelectorAll('.exchange').length > n", arg=count)
            page.wait_for_function("!document.querySelector('.progress-line') && !document.querySelector('.composer.is-busy')", timeout=240000)
            row = next(m for m in snapshot()["messages"] if m["question"] == question)
            assert row["result"].get("status") == "completed", row["result"]
            answer = row["result"]["response"]["answer"]
            fenced = re.fullmatch(r'\s*```(?:json)?\s*(\{.*\})\s*```\s*', answer, re.S)
            payload = json.loads(fenced[1] if fenced else answer)
            for key, value in expected.items():
                assert key in payload and payload[key] == value, payload
            live.append({"run_id": row["id"], "session_id": row["session_id"], "expected": expected,
                         "answer": payload, "usage": row["result"]["usage"], "citations": row["result"]["response"]["citations"], "reused_after_harness_interruption": bool(reused), "displayed_in_markdown_code_fence": bool(fenced)})
            print(json.dumps({"live_case": len(live), "passed": True, "run_id": row["id"]}), flush=True)
            return row

        try:
            page.get_by_role("button", name="Examples", exact=True).click()
            page.get_by_role("button", name="Open saved PDF example", exact=True).click()
            page.locator('.archive-label').wait_for()
            page.locator('.citation-chip').first.click()
            page.locator('.source-passage').wait_for()
            assert 'Recorded supplement cartons per day: 2' in page.locator('.source-passage').inner_text()
            assert 'No new inference' in page.locator('.archive-label').inner_text()
            page.screenshot(path=str(output / 'desktop-source-inspection.png'), full_page=True)
            checks.append('Saved example is labelled and opens an actual source passage')
            page.get_by_role('button', name='Close context panel').click()

            if args.live:
                # A new general question refers to its own earlier user context; no patient.
                new_general()
                page.get_by_role('button', name='Explore the evidence', exact=False).click()
                row = send('Find the PubMed record for the 2021 ESPEN practical guideline on Clinical Nutrition in cancer. Return a JSON object with pmid and year, and cite its actually retrieved source.', {'pmid':'33946039','year':2021})
                assert row['result']['response']['citations']
                row = send('For the same guideline we just discussed, what is its publication year? Return JSON with year. Retrieve or restore its source before citing it.', {'year':2021})
                assert row['result']['response']['citations']
                checks.append('Novel general question and conversational follow-up use real sources')

            page.get_by_role('button', name='Examples', exact=True).click()
            page.get_by_role('button', name='Continue a patient conversation', exact=False).click()
            page.wait_for_function("document.querySelector('select[aria-label=\"Patient context\"]').value.startsWith('P')")
            page.wait_for_function("!document.querySelector('select[aria-label=\"Patient context\"]').disabled")
            patient = snapshot()['session']['patient_id']
            if args.live:
                row = send('According to my registered fictional case PDF, return a JSON object with cartons_per_day and record_marker. Read the actual passage and cite it.', {'cartons_per_day':2,'record_marker':'ALPHA-NUTRITION-47'})
                assert row['result']['response']['citations']
                page.locator('.citation-chip').first.click()
                page.locator('.source-passage').wait_for()
                page.screenshot(path=str(output / 'desktop-live-pdf.png'), full_page=True)
                page.get_by_role('button', name='Close context panel').click()

            page.get_by_role('button', name='Patients', exact=False).first.click()
            page.get_by_role('button', name='Correct', exact=True).first.click()
            page.get_by_label('Value', exact=True).fill('62.1')
            page.get_by_role('button', name='Save correction', exact=True).click()
            page.wait_for_selector('dialog', state='detached')
            page.wait_for_function("document.querySelector('.fact-item')?.innerText.includes('62.1')")
            assert snapshot()['facts']['weight_kg']['value'] == 62.1
            assert any(f['superseded'] for f in snapshot()['fact_history'])
            checks.append('Explicit correction retains the superseded dated value')
            page.get_by_role('button', name='Close context panel').click()
            new_general()
            assert snapshot()['facts'] == {} and snapshot()['files'] == [] and snapshot()['session']['patient_id'] is None
            if args.live:
                send('What is my recorded weight in this conversation? No patient is selected. Return a JSON object with weight_kg, using null if unavailable.', {'weight_kg':None})
            previous = active()
            page.get_by_role('combobox', name='Patient context').select_option(patient)
            page.wait_for_function("old => localStorage.getItem('medical-copilot-conversation') !== old", arg=previous)
            page.wait_for_function("!document.querySelector('select[aria-label=\"Patient context\"]').disabled")
            assert snapshot()['facts']['weight_kg']['value'] == 62.1
            checks.append('Switching context starts a new conversation; returning restores patient records')
            if args.live:
                send('What is my most recent recorded weight now? Return a JSON object with weight_kg. Use the current patient records and do not infer an unrecorded measurement.', {'weight_kg':62.1})

            new_general()
            page.get_by_label('Select a PDF', exact=True).set_input_files(str(ROOT / 'data/evaluation/patient-demo-20261014/beta.pdf'))
            page.get_by_role('link', name='beta', exact=False).wait_for()
            assert len(snapshot()['files']) == 1 and snapshot()['session']['patient_id'] is None
            checks.append('A newly uploaded PDF belongs only to the general conversation')
            page.get_by_role('button', name='Close context panel').click()
            page.reload(wait_until='networkidle')
            page.wait_for_function("document.querySelector('.attached-pill')?.innerText.includes('1 attached')")
            checks.append('Conversation and uploaded document survive a page reload')

            page.get_by_role('button', name='Appearance & settings').click()
            page.get_by_label('Workspace name').fill('Clinical research desk')
            page.get_by_label('Accent colour', exact=True).fill('#8C1414')
            page.get_by_role('button', name='Save appearance').click()
            page.wait_for_selector('dialog', state='detached')
            page.reload(wait_until='networkidle')
            page.wait_for_function("document.title === 'Clinical research desk'")
            assert page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()").lower() == '#8c1414'
            checks.append('Appearance changes persist after reload')
            page.request.put(args.url + '/api/appearance', data=original)
            page.reload(wait_until='networkidle')

            page.set_viewport_size({'width':390,'height':844})
            page.get_by_role('button',name='Open navigation').click()
            page.get_by_role('button',name='Close navigation').wait_for()
            page.get_by_role('button',name='Close navigation').click()
            page.wait_for_function("document.querySelector('.sidebar').getBoundingClientRect().right <= 1")
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
            page.screenshot(path=str(output / 'mobile-workspace.png'), full_page=True)
            checks.append('Mobile navigation and composer fit without horizontal overflow')
            page.set_viewport_size({'width':1440,'height':980})
            page.locator('.brand').click()
            page.screenshot(path=str(output / 'desktop-final.png'), full_page=True)
            assert not errors, errors
            result = {'passed':True,'browser':'Playwright Chromium','checks':checks,'live':live,'page_errors':errors,
                      'new_live_questions':len(live),'initial_live_source_smoke_recorded_separately':True,'appearance_restored':True}
            (ROOT / 'data/checks/web-browser-20261015.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps({'passed':True,'browser_checks':len(checks),'live_questions':len(live),'page_errors':errors}),flush=True)
        finally:
            page.request.put(args.url + '/api/appearance', data=original)
            browser.close()


if __name__ == '__main__':
    main()
