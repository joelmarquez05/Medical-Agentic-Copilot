"""A small, versioned, manually curated guidance collection."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


class GuidanceLibrary:
    def __init__(self, data: dict[str, Any] | None = None):
        self.data = data if data is not None else json.loads(files("medical_copilot").joinpath("resources/oncology_guidance.json").read_text(encoding="utf-8"))
        self.sources = {source["id"]: source for source in self.data["sources"]}
        if len(self.sources) != len(self.data["sources"]):
            raise ValueError("Guidance source IDs must be unique.")
        for card in self.data["cards"]:
            if not set(card["source_ids"]).issubset(self.sources):
                raise ValueError("A guidance card refers to an unknown source.")

    def catalog(self) -> list[dict[str, str]]:
        return [{"topic": card["topic"], "description": card["description"]} for card in self.data["cards"]]

    def lookup(self, topics: list[str]) -> dict[str, Any]:
        selected = [card for card in self.data["cards"] if card["topic"] in topics]
        ids = sorted({source for card in selected for source in card["source_ids"]})
        return {"mode": "local_curated_guidance", "version": self.data["version"], "cards": selected, "sources": [self.sources[source] for source in ids], "limitation": "Manual paraphrases of selected guidance. No live paper search, systematic review, or automatic guideline update."}
