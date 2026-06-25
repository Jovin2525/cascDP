from __future__ import annotations
import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Set, Tuple
import numpy as np
from src.evaluation.caid_io import merge_submission_files

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAID_DIR = _REPO_ROOT / "third_party" / "caid"

BINDING_TYPES = (
    ("protein", "PROTEIN", 0),
    ("nucleic", "NUCLEIC", 1),
    ("ion", "ION", 2),
    ("lipid", "LIPID", 3),
)

@dataclass(frozen=True)
class CaidTask:
    name: str
    score_key: str
    state_key: str
    reference_path: Optional[Path] = None

CAID3_TASKS: Mapping[str, CaidTask] = {
    "caid3_disorder_nox": CaidTask(
        name="disorder_nox",
        score_key="disorder_probs",
        state_key="disorder_pred",
        reference_path=_REPO_ROOT / "data/eval/caid3/disorder_nox_caid3.fasta",
    ),
    "caid3_disorder_pdb": CaidTask(
        name="disorder_pdb",
        score_key="disorder_probs",
        state_key="disorder_pred",
        reference_path=_REPO_ROOT / "data/eval/caid3/disorder_pdb_caid3.fasta",
    ),
    "caid3_binding": CaidTask(
        name="binding",
        score_key="binding_probs_combined",
        state_key="binding_pred",
        reference_path=_REPO_ROOT / "data/eval/caid3/binding_caid3.fasta",
    ),
    "caid3_binding_idr": CaidTask(
        name="binding_idr",
        score_key="binding_probs_combined",
        state_key="binding_pred",
        reference_path=_REPO_ROOT / "data/eval/caid3/binding_idr_caid3.fasta",
    ),
    "caid3_linker": CaidTask(
        name="linker",
        score_key="linker_probs",
        state_key="linker_pred",
        reference_path=_REPO_ROOT / "data/eval/caid3/linker_caid3.fasta",
    ),
}

TEST_FINAL_TASKS: Tuple[CaidTask, ...] = (
    CaidTask(name="disorder", score_key="disorder_probs", state_key="disorder_pred"),
    CaidTask(name="binding", score_key="binding_probs_combined", state_key="binding_pred"),
    CaidTask(name="linker", score_key="linker_probs", state_key="linker_pred"),
)


def load_bvaluation():
    # BioComputingUP/CAID uses np.trapz; NumPy 2.x exposes the same function as np.trapezoid.
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid

    if not _CAID_DIR.exists():
        raise RuntimeError(f"vendored CAID directory not found: {_CAID_DIR}")
    caid_path = str(_CAID_DIR)
    if caid_path not in sys.path:
        sys.path.insert(0, caid_path)
    try:
        from vectorized_metrics.vectorized_metrics import bvaluation
    except ImportError as exc:
        raise RuntimeError(
            "could not import vendored CAID evaluator; ensure numpy, pandas, scipy, and tqdm "
            "are installed in the evaluation environment"
        ) from exc
    return bvaluation

def submission_flavors(test_set: str, selected_flavors: Optional[Set[str]]):
    if test_set == "caid3_disorder_nox":
        candidates = [("disorder_nox", "disorder_probs", "disorder_pred")]
    elif test_set == "caid3_disorder_pdb":
        candidates = [("disorder_pdb", "disorder_probs", "disorder_pred")]
    elif test_set == "caid3_binding":
        candidates = [("binding", "binding_probs_combined", "binding_pred")]
    elif test_set == "caid3_binding_idr":
        candidates = [("binding_idr", "binding_probs_combined", "binding_pred")]
    elif test_set == "caid3_linker":
        candidates = [("linker", "linker_probs", "linker_pred")]
    else:
        candidates = [
            ("disorder", "disorder_probs", "disorder_pred"),
            ("binding", "binding_probs_combined", "binding_pred"),
            ("binding_protein", "binding_probs_protein", "binding_protein_pred"),
            ("binding_nucleic", "binding_probs_nucleic", "binding_nucleic_pred"),
            ("binding_ion", "binding_probs_ion", "binding_ion_pred"),
            ("binding_lipid", "binding_probs_lipid", "binding_lipid_pred"),
            ("linker", "linker_probs", "linker_pred"),
        ]

    if selected_flavors is None:
        return candidates

    selected = []
    for flavor, score_key, state_key in candidates:
        if flavor.startswith("disorder"):
            base = "disorder"
        elif flavor.startswith("binding"):
            base = "binding"
        else:
            base = flavor
        if flavor in selected_flavors or base in selected_flavors:
            selected.append((flavor, score_key, state_key))
    return selected

def run_caid_metrics(
    dataset,
    test_set: str,
    submission_dir: Path,
    metrics_root: Path,
    merged_dir: Path,
    reference_dir: Path,
    selected_flavors: Optional[Set[str]],
    produced_flavors: Set[str],
    skip_bootstrap: bool,
):
    bvaluation = load_bvaluation()
    specs = metric_specs(test_set, dataset, reference_dir, selected_flavors)
    completed = []
    for spec in specs:
        name = spec["name"]
        flavor = spec["flavor"]
        if flavor not in produced_flavors:
            logger.warning(
                "Skipping CAID metrics for %s: flavor '%s' was not produced in this run",
                name,
                flavor,
            )
            continue
        pids = spec.get("pids")
        reference = spec["reference"]
        prediction = merged_dir / f"cascdp_{test_set}_{name}.caid"
        rows = merge_submission_files(submission_dir / flavor, prediction, pids)
        if rows == 0:
            logger.warning("Skipping CAID metrics for %s: no prediction rows", name)
            continue

        outpath = metrics_root / name
        outpath.mkdir(parents=True, exist_ok=True)
        np.random.seed(42)
        bvaluation(
            reference=reference,
            predictions=[prediction],
            outpath=outpath,
            dataset=True,
            target=True,
            bootstrap=not skip_bootstrap,
            accs_to_read=None,
        )
        logger.info("CAID metrics complete for %s: %s", name, outpath)
        completed.append({**spec, "outpath": outpath})
    return completed

def metric_specs(test_set: str, dataset, reference_dir: Path, selected_flavors: Optional[Set[str]]):
    if test_set in CAID3_TASKS:
        task = CAID3_TASKS[test_set]
        if selected_flavors is not None:
            if task.name.startswith("disorder"):
                base = "disorder"
            elif task.name.startswith("binding"):
                base = "binding"
            else:
                base = task.name
            if task.name not in selected_flavors and base not in selected_flavors:
                return []
        return [
            {
                "name": task.name,
                "flavor": task.name,
                "reference": task.reference_path,
            }
        ]

    sequences = getattr(dataset, "sequences", {}) or {}
    source_lookup = getattr(dataset, "protein_sources", {}) or {}
    all_pids = sorted(sequences)

    specs = []
    enabled = selected_flavors is None or "disorder" in selected_flavors
    if enabled:
        specs.append(
            {
                "name": "disorder_all",
                "flavor": "disorder",
                "reference": write_reference(dataset, reference_dir, "disorder_all", "disorder", all_pids),
                "pids": set(all_pids),
            }
        )
        disprot_pids = [pid for pid in all_pids if "PDB" not in source_lookup.get(pid, "DisProt")]
        pdb_pids = [pid for pid in all_pids if "PDB" in source_lookup.get(pid, "DisProt")]
        if disprot_pids:
            specs.append(
                {
                    "name": "disorder_disprot",
                    "flavor": "disorder",
                    "reference": write_reference(dataset, reference_dir, "disorder_disprot", "disorder", disprot_pids),
                    "pids": set(disprot_pids),
                }
            )
        else:
            logger.warning("Skipping disorder_disprot metrics: no DisProt-sourced proteins found")
        if pdb_pids:
            specs.append(
                {
                    "name": "disorder_pdb",
                    "flavor": "disorder",
                    "reference": write_reference(dataset, reference_dir, "disorder_pdb", "disorder", pdb_pids),
                    "pids": set(pdb_pids),
                }
            )
        else:
            logger.warning("Skipping disorder_pdb metrics: no PDB-sourced proteins found")

    binding_enabled = selected_flavors is None or any(
        flavor == "binding" or flavor.startswith("binding_") for flavor in selected_flavors
    )
    if binding_enabled:
        binding_pids = positive_task_pids(dataset, all_pids, "binding")
        if binding_pids:
            specs.append(
                {
                    "name": "binding",
                    "flavor": "binding",
                    "reference": write_reference(dataset, reference_dir, "binding", "binding", binding_pids),
                    "pids": set(binding_pids),
                }
            )
        else:
            logger.warning("Skipping binding metrics: no proteins with positive binding labels found")
        for type_name, _type_label, _type_idx in BINDING_TYPES:
            task = f"binding_{type_name}"
            task_pids = positive_task_pids(dataset, all_pids, task)
            if task_pids:
                specs.append(
                    {
                        "name": task,
                        "flavor": task,
                        "reference": write_reference(dataset, reference_dir, task, task, task_pids),
                        "pids": set(task_pids),
                    }
                )
            else:
                logger.warning("Skipping %s metrics: no proteins with positive labels found", task)
    if selected_flavors is None or "linker" in selected_flavors:
        linker_pids = positive_task_pids(dataset, all_pids, "linker")
        if linker_pids:
            specs.append(
                {
                    "name": "linker",
                    "flavor": "linker",
                    "reference": write_reference(dataset, reference_dir, "linker", "linker", linker_pids),
                    "pids": set(linker_pids),
                }
            )
        else:
            logger.warning("Skipping linker metrics: no proteins with positive linker labels found")
    return specs

def write_reference(dataset, reference_dir: Path, name: str, task: str, pids: Iterable[str]) -> Path:
    reference_dir.mkdir(parents=True, exist_ok=True)
    path = reference_dir / f"{name}.fasta"
    sequences = getattr(dataset, "sequences", {}) or {}
    with open(path, "w") as handle:
        for pid in sorted(pids):
            seq = sequences.get(pid)
            labels, mask = labels_and_mask(dataset, pid, task)
            if seq is None or labels is None or mask is None:
                continue
            length = min(len(seq), len(labels), len(mask))
            label_line = "".join(reference_char(labels[i], mask[i]) for i in range(length))
            handle.write(f">{pid}\n{seq[:length]}\n{label_line}\n")
    return path

def labels_and_mask(dataset, pid: str, task: str):
    if task == "disorder":
        return getattr(dataset, "disorder_labels", {}).get(pid, (None, None))
    function_labels = getattr(dataset, "function_labels", {}).get(pid)
    if function_labels is None:
        return None, None
    if task == "binding":
        return function_labels[:, 4], getattr(dataset, "binding_masks", {}).get(pid)
    for type_name, _type_label, type_idx in BINDING_TYPES:
        if task == f"binding_{type_name}":
            type_masks = getattr(dataset, "binding_type_masks", {}).get(pid)
            mask = type_masks[:, type_idx] if type_masks is not None else getattr(dataset, "binding_masks", {}).get(pid)
            return function_labels[:, type_idx], mask
    if task == "linker":
        return function_labels[:, 5], getattr(dataset, "linker_masks", {}).get(pid)
    return None, None


def positive_task_pids(dataset, pids: Iterable[str], task: str):
    positive_pids = []
    for pid in pids:
        labels, mask = labels_and_mask(dataset, pid, task)
        if labels is None or mask is None:
            continue
        length = min(len(labels), len(mask))
        if not length:
            continue
        labels_arr = np.asarray(labels[:length])
        mask_arr = np.asarray(mask[:length])
        if np.any((mask_arr >= 0.5) & (labels_arr >= 0.5)):
            positive_pids.append(pid)
    return positive_pids

def reference_char(label: float, mask: float) -> str:
    if float(mask) < 0.5:
        return "-"
    return "1" if float(label) >= 0.5 else "0"

def format_caid_metrics_summary(completed_specs) -> str:
    if not completed_specs:
        return ""

    by_name = {spec["name"]: spec for spec in completed_specs}
    lines = []
    lines.append("=" * 60)
    lines.append("OFFICIAL CAID METRICS")
    lines.append("Metrics are read from CAID bvaluation CSV outputs.")
    lines.append("Pool: dataset-level (micro). per-target avg = mean over CAID target rows.")
    lines.append("=" * 60)

    def append_task(title: str, name: str, full: bool = True) -> bool:
        spec = by_name.get(name)
        if spec is None:
            return False
        dataset_row = read_single_caid_metrics(spec["outpath"], "dataset")
        if not dataset_row:
            return False
        if title:
            lines.append(f"\n{title}:")
            lines.append("-" * 60)
        append_metric_block(lines, dataset_row, full=full)
        target_summary = read_target_caid_metrics(spec["outpath"])
        if target_summary:
            lines.append(
                f"  -- per-target avg ({target_summary['n_targets']:.0f} proteins, "
                f"thr={format_metric(dataset_row.get('thr'), 3)}) --"
            )
            lines.append(f"  F1:        {format_metric(target_summary.get('f1s'))}")
            lines.append(f"  Precision: {format_metric(target_summary.get('ppv'))}")
            lines.append(f"  Recall:    {format_metric(target_summary.get('tpr'))}")
            lines.append(f"  MCC:       {format_metric(target_summary.get('mcc'))}")
            lines.append(f"  BAC:       {format_metric(target_summary.get('bac'))}")
        return True

    disorder_names = ["disorder_all", "disorder", "disorder_nox", "disorder_pdb"]
    for name in disorder_names:
        if append_task("DISORDER PREDICTION", name):
            break

    for name, label in (("disorder_disprot", "DisProt subset"), ("disorder_pdb", "PDB_missing subset")):
        if name in by_name and "disorder_all" in by_name:
            lines.append(f"\n  -- {label} --")
            spec = by_name[name]
            row = read_single_caid_metrics(spec["outpath"], "dataset")
            if row:
                append_metric_block(lines, row, full=False)

    if append_task("BINDING PREDICTION (Combined: Protein/Nucleic/Ion/Lipid)", "binding"):
        for type_name, type_label, _type_idx in BINDING_TYPES:
            name = f"binding_{type_name}"
            if name in by_name:
                lines.append(f"\n  -- {type_label} BINDING --")
                row = read_single_caid_metrics(by_name[name]["outpath"], "dataset")
                if row:
                    append_metric_block(lines, row, full=False)

    append_task("BINDING IDR PREDICTION", "binding_idr")
    append_task("LINKER PREDICTION", "linker")

    lines.append("=" * 60)
    return "\n".join(lines)

def append_metric_block(lines, row, full: bool) -> None:
    lines.append(f"  ROC-AUC:   {format_metric(row.get('aucroc'))}")
    lines.append(f"  PR-AUC:    {format_metric(row.get('aucpr', row.get('aps')))}")
    lines.append(f"  APS:       {format_metric(row.get('aps'))}")
    lines.append(f"  F_max:     {format_metric(row.get('f1s'))}")
    if full:
        lines.append(f"  Threshold: {format_metric(row.get('thr'))}")
        lines.append(f"  Precision: {format_metric(row.get('ppv'))}")
        lines.append(f"  Recall:    {format_metric(row.get('tpr'))}")
    lines.append(f"  MCC:       {format_metric(row.get('mcc'))}")
    lines.append(f"  BAC:       {format_metric(row.get('bac'))}")

def read_single_caid_metrics(outpath: Path, scope: str):
    pattern = f"*.analysis.all.{scope}.f1s.metrics.csv"
    paths = sorted(outpath.glob(pattern))
    if not paths:
        return {}
    with open(paths[0], newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
            values = next(reader)
        except StopIteration:
            return {}
    return metrics_row_to_dict(header, values, skip_columns=1)

def read_target_caid_metrics(outpath: Path):
    paths = sorted(outpath.glob("*.analysis.all.target.f1s.metrics.csv"))
    if not paths:
        return {}
    sums = {}
    counts = {}
    n_targets = 0
    with open(paths[0], newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {}
        for values in reader:
            if not values:
                continue
            n_targets += 1
            row = metrics_row_to_dict(header, values, skip_columns=2)
            for key in ("f1s", "ppv", "tpr", "mcc", "bac"):
                value = parse_float(row.get(key))
                if value is None:
                    continue
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
    if n_targets == 0:
        return {}
    summary = {key: sums[key] / counts[key] for key in sums if counts.get(key)}
    summary["n_targets"] = float(n_targets)
    return summary

def metrics_row_to_dict(header, values, skip_columns: int):
    row = {}
    for key, value in zip(header[skip_columns:], values[skip_columns:]):
        if not key:
            continue
        row[key] = value
    return row

def format_metric(value, places: int = 4) -> str:
    numeric = parse_float(value)
    if numeric is None:
        return "  n/a "
    return f"{numeric:.{places}f}"

def parse_float(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric