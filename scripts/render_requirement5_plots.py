"""Export descriptive comparison figures with standard Matplotlib tools.

Run with a Python environment containing Matplotlib; no model/API call occurs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/comparison-runs/selected-five-comparison-20261010"
OUT = ROOT / "docs/implementation/figures"
NAMES = {"openrouter-claude-opus-5.5": "Claude Opus 5.5", "openrouter-glm-5.3": "GLM-5.3", "openrouter-qwen3.8-27b": "Qwen3.8 27B", "gemini-3.8-flash": "Gemini 3.8 Flash", "gemma-4-31b": "Gemma 4 31B"}
COLORS = ["#225b88", "#307f74", "#8963a1", "#c88a30", "#6785ab"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def export(fig, name):
    for extension in ["pdf", "svg"]:
        fig.savefig(OUT / (name + "." + extension), bbox_inches="tight", metadata={"Creator": "Matplotlib descriptive internal evaluation"} if extension == "pdf" else None)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    quality = read(RUN / "qualitative-summary.json")["models"]
    operations = read(RUN / "engineering-summary.json")["models"]
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    x = np.arange(len(quality))
    labels = [NAMES[model["profile"]] for model in quality]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for index, model in enumerate(quality):
        mean = model["mean_ordinal_total"]
        if mean is not None:
            axes[0].bar(index, mean, color=COLORS[index], width=0.65)
            axes[0].text(index, mean + 0.12, f"{mean:.1f}\n(n={model['scored_final_answers']})", ha="center", va="bottom", fontsize=9)
        else:
            axes[0].text(index, 0.5, "N/A\nno final answers", ha="center", va="bottom", fontsize=9)
        axes[1].bar(index, operations[index]["completed_answers"], color=COLORS[index], width=0.65)
        axes[1].text(index, operations[index]["completed_answers"] + 0.12, f"{operations[index]['completed_answers']}/10", ha="center", va="bottom", fontsize=9)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(0, 11)
        ax.set_yticks(range(0, 11, 2))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean ordinal checklist total (0-10)")
    axes[0].set_title("Saved final-answer assessment")
    axes[1].set_ylabel("Completed final answers")
    axes[1].set_title("Coverage of the ten-question internal set")
    fig.suptitle("First completed answer per case; unblinded review; not medical accuracy", fontsize=11)
    export(fig, "requirement-5-quality-and-completion")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for index, model in enumerate(operations):
        latency = model["median_case_elapsed_seconds"]
        if latency is not None:
            axes[0].bar(index, latency, color=COLORS[index], width=0.65)
            axes[0].text(index, latency + 3, f"{latency:.0f}", ha="center", va="bottom", fontsize=9)
        cost = model["reported_or_estimated_cost_usd"]
        if cost is not None:
            axes[1].bar(index, cost, color=COLORS[index], width=0.65)
            axes[1].text(index, cost + 0.03, f"${cost:.3f}", ha="center", va="bottom", fontsize=9)
        else:
            axes[1].text(index, 0.04, "Cost not\nreported", ha="center", va="bottom", fontsize=9)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.margins(y=0.25)
    axes[0].set_ylabel("Median agent-case elapsed seconds")
    axes[0].set_title("Selected graphs; earlier failures excluded")
    axes[1].set_ylabel("Main-cohort OpenRouter usage cost (USD)")
    axes[1].set_title("API reports; Google cost is unknown")
    fig.suptitle("Google recovery adds retries; latency includes stages/pacing; pilot cost excluded", fontsize=11)
    export(fig, "requirement-5-latency-and-cost")
    metadata = {"matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__, "source_files": ["engineering-summary.json", "qualitative-summary.json"], "interpretation": "Descriptive ordinal/operational figures only, with N/A scores and unknown Google costs explicit. No uncertainty intervals or statistical superiority claims.", "exports": ["requirement-5-quality-and-completion.pdf", "requirement-5-quality-and-completion.svg", "requirement-5-latency-and-cost.pdf", "requirement-5-latency-and-cost.svg"]}
    (OUT / "requirement-5-plots-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Exported four standalone comparison figures.")


if __name__ == "__main__":
    main()
