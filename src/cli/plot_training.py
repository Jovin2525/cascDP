"""
Plot training metrics from training_history.json

Usage:
    python -m src.cli.plot_training --history checkpoints/phase1/training_history.json
"""

import argparse
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BINDING_TYPES = ['protein', 'nucleic_acid', 'ion', 'lipid']
TASK_METRICS = ['f_max', 'auc', 'aps']

def _has_data(metrics_list, key):
    """Return True if `key` exists with at least one finite, non-zero value."""
    if not metrics_list:
        return False
    for entry in metrics_list:
        if not isinstance(entry, dict):
            continue
        if key not in entry:
            continue
        val = entry[key]
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(v) or v == 0.0:
            continue
        return True
    return False

def _train_vals_have_signal(train_vals):
    """Return True if any train value is non-zero (non-degenerate panel)."""
    return any(v != 0.0 and not (isinstance(v, float) and math.isnan(v)) for v in train_vals)

def _extract_series(metrics_list, key):
    """Extract a list of float values for `key` from each val_metrics entry."""
    series = []
    for entry in metrics_list:
        if isinstance(entry, dict) and key in entry:
            try:
                v = float(entry[key])
            except (TypeError, ValueError):
                v = float('nan')
        else:
            v = float('nan')
        series.append(v)
    return series
def _plot_train_val(ax, train_vals, val_vals, title, train_label='Train', val_label='Validation'):
    """Plot train (solid) and val (dashed) series on a single axis with best-value annotation."""
    train_epochs = range(1, len(train_vals) + 1)
    ax.plot(train_epochs, train_vals, marker='o', linewidth=2,
            color='steelblue', label=train_label)

    if val_vals and len(val_vals) > 0:
        val_epochs = range(1, len(val_vals) + 1)
        finite_val = [v for v in val_vals if not (isinstance(v, float) and math.isnan(v))]
        if finite_val:
            ax.plot(val_epochs, val_vals, marker='s', linewidth=2,
                    color='darkorange', linestyle='--', label=val_label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss' if 'loss' in title.lower() else 'Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_metric_panel(ax, values, epochs, title):
    """Plot a single validation metric with best-value annotation."""
    finite_vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not finite_vals:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.plot(epochs, values, marker='o', color='steelblue', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_losses(history, output_dir):
    """2x2 grid: total/disorder/binding/linker losses with train and val overlays."""
    train_total = history.get('train_loss', [])
    val_total = history.get('val_loss', [])

    train_disorder = history.get('train_disorder_loss', [])
    train_binding = history.get('train_binding_loss', [])
    train_linker = history.get('train_linker_loss', [])

    val_metrics = history.get('val_metrics', [])
    val_disorder = _extract_series(val_metrics, 'val_disorder_loss')
    val_binding = _extract_series(val_metrics, 'val_binding_loss')
    val_linker = _extract_series(val_metrics, 'val_linker_loss')

    has_disorder = _train_vals_have_signal(train_disorder) or _has_data(val_metrics, 'val_disorder_loss')
    has_binding = _train_vals_have_signal(train_binding) or _has_data(val_metrics, 'val_binding_loss')
    has_linker = _train_vals_have_signal(train_linker) or _has_data(val_metrics, 'val_linker_loss')

    panels = [('Total Loss', train_total, val_total)]
    if has_disorder:
        panels.append(('Disorder Loss', train_disorder, val_disorder))
    if has_binding:
        panels.append(('Binding Loss', train_binding, val_binding))
    if has_linker:
        panels.append(('Linker Loss', train_linker, val_linker))

    n = len(panels)
    if n == 0:
        print("  Skipping losses.png: no loss data in history")
        return

    if n <= 2:
        rows, cols = 1, n
    else:
        rows, cols = 2, 2
        if n == 3:
            rows, cols = 1, 3
        else:
            rows, cols = 2, 2

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    if n == 1:
        axes_list = [axes]
    else:
        axes_list = list(np.atleast_1d(axes).flatten())

    for ax, (title, t_vals, v_vals) in zip(axes_list, panels):
        _plot_train_val(ax, t_vals, v_vals, title)

    for ax in axes_list[len(panels):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'losses.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'losses.png'}")
    plt.close(fig)

def plot_disorder_metrics(history, output_dir):
    """1x3: disorder F_max / AUC / APS across validation epochs."""
    val_metrics = history.get('val_metrics', [])
    if not val_metrics:
        print("  Skipping disorder_metrics.png: no val_metrics")
        return

    keys = ['disorder_f_max', 'disorder_auc', 'disorder_aps']
    titles = ['Disorder F_max', 'Disorder AUC', 'Disorder APS']

    if not any(_has_data(val_metrics, k) for k in keys):
        print("  Skipping disorder_metrics.png: no disorder metrics in val_metrics")
        return

    epochs = list(range(1, len(val_metrics) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, title in zip(axes, keys, titles):
        values = _extract_series(val_metrics, key)
        if epochs:
            ax.set_xlim(0.5, len(epochs) + 0.5)
        _plot_metric_panel(ax, values, epochs, title)

    plt.tight_layout()
    plt.savefig(output_dir / 'disorder_metrics.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'disorder_metrics.png'}")
    plt.close(fig)

def plot_function_metrics(history, output_dir):
    """Grid of binding and/or linker rows x F_max/AUC/APS columns."""
    val_metrics = history.get('val_metrics', [])
    if not val_metrics:
        print("  Skipping function_metrics.png: no val_metrics")
        return

    task_keys = {
        'Binding': ['binding_f_max', 'binding_auc', 'binding_aps'],
        'Linker':  ['linker_f_max', 'linker_auc', 'linker_aps'],
    }

    present_tasks = [
        (name, keys) for name, keys in task_keys.items()
        if any(_has_data(val_metrics, k) for k in keys)
    ]

    if not present_tasks:
        print("  Skipping function_metrics.png: no function metrics in val_metrics")
        return

    rows = len(present_tasks)
    cols = len(TASK_METRICS)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    epochs = list(range(1, len(val_metrics) + 1))
    for row_idx, (task_name, keys) in enumerate(present_tasks):
        for col_idx, (key, metric_name) in enumerate(zip(keys, TASK_METRICS)):
            ax = axes[row_idx, col_idx]
            values = _extract_series(val_metrics, key)
            if epochs:
                ax.set_xlim(0.5, len(epochs) + 0.5)
            _plot_metric_panel(ax, values, epochs, f'{task_name} {metric_name.upper()}')

    plt.tight_layout()
    plt.savefig(output_dir / 'function_metrics.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'function_metrics.png'}")
    plt.close(fig)

def plot_per_binding_type_metrics(history, output_dir):
    """4x3 grid: per binding-type F_max / AUC / APS."""
    val_metrics = history.get('val_metrics', [])
    if not val_metrics:
        print("  Skipping per_binding_type_metrics.png: no val_metrics")
        return

    type_keys = {btype: f'binding_{btype}' for btype in BINDING_TYPES}
    metrics_keys = ['f_max', 'auc', 'aps']

    present = [btype for btype, prefix in type_keys.items()
               if any(_has_data(val_metrics, f'{prefix}_{mk}') for mk in metrics_keys)]
    if not present:
        print("  Skipping per_binding_type_metrics.png: no per-binding-type metrics")
        return

    rows = len(present)
    cols = len(metrics_keys)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    epochs = list(range(1, len(val_metrics) + 1))
    for row_idx, btype in enumerate(present):
        prefix = type_keys[btype]
        for col_idx, mk in enumerate(metrics_keys):
            ax = axes[row_idx, col_idx]
            values = _extract_series(val_metrics, f'{prefix}_{mk}')
            if epochs:
                ax.set_xlim(0.5, len(epochs) + 0.5)
            _plot_metric_panel(ax, values, epochs, f'{btype} {mk.upper()}')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_binding_type_metrics.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'per_binding_type_metrics.png'}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--history', type=str, default='checkpoints/training_history.json',
                        help='Path to training_history.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as history file)')
    parser.add_argument('--no-per-binding-type', action='store_true',
                        help='Skip the per-binding-type metrics plot')
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        print(f"Error: {history_path} not found!")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = history_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots from {history_path}...")
    print(f"Output directory: {output_dir}")

    plot_losses(history, output_dir)
    plot_disorder_metrics(history, output_dir)
    plot_function_metrics(history, output_dir)
    if not args.no_per_binding_type:
        plot_per_binding_type_metrics(history, output_dir)

    print("\nAll plots generated successfully!")
    print(f"Files saved in: {output_dir}")

if __name__ == '__main__':
    main()