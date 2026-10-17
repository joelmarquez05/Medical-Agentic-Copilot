"""Typed state decisions and tool arguments; no arbitrary tool execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Topic = Literal["screening", "requirements", "enteral", "parenteral", "cachexia", "refeeding", "food_safety", "supplements", "pancreatic_enzymes", "restrictive_diets"]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GuidanceArgs(StrictSchema):
    topics: list[Topic] = Field(min_length=1, max_length=6)


class TargetArgs(StrictSchema):
    weight_kg: float = Field(gt=0, le=500, allow_inf_nan=False, strict=True)


class ToolCall(StrictSchema):
    name: Literal["lookup_guidance", "estimate_nutrition_targets"]
    arguments: dict[str, Any]


class Plan(StrictSchema):
    scope: Literal["oncology_nutrition", "out_of_scope"]
    answer_focus: list[str] = Field(max_length=6)
    missing_information: list[str] = Field(max_length=6)
    tool_calls: list[ToolCall] = Field(max_length=4)


class Review(StrictSchema):
    needs_revision: bool
    issues: list[str] = Field(max_length=8)
    additional_topics: list[Topic] = Field(max_length=3)
    uncertainty_note: str
