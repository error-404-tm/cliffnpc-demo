"""Shared helpers for the experiment scripts (headless: plots go to files)."""

from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DRIVE_LABELS = ["hunger e1", "social e2", "explore e3", "threat e4"]
DRIVE_COLORS = ["#c0392b", "#2e86c1", "#27ae60", "#8e44ad"]


def pmap(fn, items, workers: int | None = None) -> list:
    """Map `fn` over independent runs in parallel processes, results in order.

    Runs are seeded, so the results are identical to a serial loop.  Uses the
    "fork" start method explicitly: Python 3.14 defaults to "forkserver", which
    would re-execute the experiment script in every worker.  Set
    CLIFFNPC_WORKERS=1 to run serially (e.g. for debugging).
    """
    items = list(items)
    if workers is None:
        workers = int(os.environ.get("CLIFFNPC_WORKERS", os.cpu_count() or 1))
    workers = max(1, min(workers, len(items)))
    if workers == 1:
        return [fn(x) for x in items]
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        return list(ex.map(fn, items))


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
