from __future__ import annotations
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Set

logger = logging.getLogger(__name__)

SUBMISSION_TASKS = (
    ("disorder", "disorder_probs", "disorder_pred"),
    ("binding", "binding_probs_combined", "binding_pred"),
    ("linker", "linker_probs", "linker_pred"),
)

def write_caid_file(
    path: Path,
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    score_key: str,
    state_key: str,
    precision: int = 6,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0

    with open(path, "w") as handle:
        for protein_id in sorted(predictions):
            prediction = predictions[protein_id]
            sequence = prediction.get("sequence")
            scores = prediction.get(score_key)
            states = prediction.get(state_key)
            if not sequence or scores is None:
                continue
            if states is None:
                logger.warning(
                    "Skipping %s for %s: scores exist but no binary states are available",
                    score_key,
                    protein_id,
                )
                continue

            length = min(len(sequence), len(scores), len(states))
            if length == 0:
                continue

            handle.write(f">{protein_id}\n")
            for position, (aa, score, state) in enumerate(
                zip(sequence[:length], scores[:length], states[:length]),
                start=1,
            ):
                handle.write(f"{position}\t{aa}\t{float(score):.{precision}f}\t{int(state)}\n")
                n_rows += 1

    return n_rows

def write_per_protein_caid(
    path: Path,
    pid: str,
    sequence: str,
    scores,
    states,
    precision: int = 3,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    length = min(len(sequence), len(scores), len(states))
    with open(path, "w") as handle:
        handle.write(f">{pid}\n")
        for pos, (aa, score, state) in enumerate(
            zip(sequence[:length], scores[:length], states[:length]),
            start=1,
        ):
            handle.write(f"{pos}\t{aa}\t{float(score):.{precision}f}\t{int(state)}\n")

def write_timings(path: Path, timings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        started = datetime.now().astimezone().strftime("%a %b %d %H:%M:%S %Z %Y")
        handle.write(f"# Running cascDP, started {started}\n")
        writer = csv.writer(handle)
        writer.writerow(["sequence", "milliseconds"])
        writer.writerows(timings)
    logger.info("Saved timings: %s", path)

def merge_submission_files(flavor_dir: Path, merged_path: Path, pids: Optional[Set[str]]) -> int:
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    safe_pids = {safe_filename(p) for p in pids} if pids is not None else None
    rows = 0
    with open(merged_path, "w") as out:
        for path in sorted(flavor_dir.glob("*.caid")):
            pid = path.stem
            if safe_pids is not None and pid not in safe_pids:
                continue
            text = path.read_text()
            out.write(text)
            if text and not text.endswith("\n"):
                out.write("\n")
            rows += sum(1 for line in text.splitlines() if line and not line.startswith(">"))
    return rows

def safe_filename(pid: str) -> str:
    return pid.replace("/", "_").replace("\\", "_")
