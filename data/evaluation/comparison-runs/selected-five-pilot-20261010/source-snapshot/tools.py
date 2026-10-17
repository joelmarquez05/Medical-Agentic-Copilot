"""Read-only local guidance and deterministic adult nutrition calculations."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from medical_copilot.evidence import GuidanceLibrary
from medical_copilot.schemas import GuidanceArgs, TargetArgs


def estimate_nutrition_targets(weight_kg: float) -> dict[str, Any]:
    args = TargetArgs(weight_kg=weight_kg)
    weight = args.weight_kg
    return {"weight_kg": weight, "energy_kcal_day": {"lower": round(weight * 25, 1), "upper": round(weight * 30, 1)}, "protein_g_day": {"greater_than": round(weight * 1.0, 1), "up_to": round(weight * 1.5, 1)}, "formula": "Energy: weight * 25 to weight * 30 kcal/day. Protein: greater than weight * 1.0, if possible up to weight * 1.5 g/day.", "source_ids": ["S01"], "limitation": "General adult estimates from ESPEN, not a prescription. Do not use as an initial refeeding rate. Individual assessment is needed for renal/organ dysfunction, obesity, edema, and other special circumstances."}


def make_tools(library: GuidanceLibrary) -> dict[str, StructuredTool]:
    def lookup_guidance(topics: list[str]) -> dict[str, Any]:
        validated = GuidanceArgs(topics=topics)
        return library.lookup(validated.topics)

    return {
        "lookup_guidance": StructuredTool.from_function(func=lookup_guidance, name="lookup_guidance", description="Read manually curated oncology-nutrition guidance by topic. This is not PubMed or medRxiv search.", args_schema=GuidanceArgs),
        "estimate_nutrition_targets": StructuredTool.from_function(func=estimate_nutrition_targets, name="estimate_nutrition_targets", description="Calculate general adult cancer energy and protein estimates from an explicitly supplied weight in kilograms. Not for initiating refeeding.", args_schema=TargetArgs),
    }
