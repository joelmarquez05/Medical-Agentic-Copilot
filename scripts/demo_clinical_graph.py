"""Exercise the new graph with clearly labeled local synthetic fixtures only."""

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from medical_copilot.clinical_graph import ClinicalAgent, EvidenceRecord, ToolDefinition, ToolOutput, save_trace


QUESTION = {"question_id": "SYNTHETIC-DEMO", "question": "Which color marker is in the synthetic document?",
            "options": {"A": "BLUE", "B": "GREEN"}}


class LookupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


class SyntheticModel:
    def __init__(self, use_tool):
        self.use_tool = use_tool

    def reply(self, request):
        if self.use_tool and not request["history"]:
            return {"type": "tool", "name": "synthetic_lookup", "arguments": {"query": "color marker"}}
        found = "SYNTHETIC_COLOR_DOCUMENT" in request["evidence_registry"]
        return {"type": "final", "answer": "BLUE", "selected_option": "A",
                "justification": "This local fixture demonstrates result feedback and routing; it is not a medical evaluation.",
                "citations": ["SYNTHETIC_COLOR_DOCUMENT"] if found else [],
                "limitations": ["Every model response and source in this demo is synthetic."]}


def lookup(args):
    return ToolOutput(content="SIMULATED local lookup succeeded.", sources=[EvidenceRecord(
        id="SYNTHETIC_COLOR_DOCUMENT", title="Synthetic color document, not a real publication",
        text="SIMULATED DOCUMENT: the marker is BLUE.", text_kind="synthetic",
    )])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing artifacts cannot be overwritten.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline = ClinicalAgent(SyntheticModel(False))
    tool_agent = ClinicalAgent(SyntheticModel(True), [ToolDefinition(
        "synthetic_lookup", "Synthetic local execution fixture; no PubMed or network.", LookupArgs, lookup,
    )])
    for label, agent in [("baseline", baseline), ("tool-loop", tool_agent)]:
        result = agent.run(QUESTION)
        result["demonstration"] = {"kind": "synthetic_offline", "network_requests": 0,
                                   "hosted_model_requests": 0, "medical_performance_evaluated": False}
        save_trace(args.output_dir / f"{label}.json", result)
        print(f"{label}: {result['status']}; model fixture turns={result['model_turns']}; local tool actions={result['tool_actions']}")
    (args.output_dir / "graph.mmd").write_text(tool_agent.graph.get_graph().draw_mermaid(), encoding="utf-8")
    print("All responses/sources are synthetic. No credentials, network or hosted inference used.")


if __name__ == "__main__":
    main()
