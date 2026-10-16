"""Shared helpers for the experiment scripts (headless: plots go to files)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DRIVE_LABELS = ["hunger e1", "social e2", "explore e3", "threat e4"]
DRIVE_COLORS = ["#c0392b", "#2e86c1", "#27ae60", "#8e44ad"]


def outdir(name: str) -> Path:
    d = ROOT / "experiments" / "out" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, d: Path, fname: str) -> None:
    fig.tight_layout()
    fig.savefig(d / fname, dpi=130)
    plt.close(fig)


def write_summary(d: Path, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    (d / "summary.txt").write_text(text)
    print(text)
