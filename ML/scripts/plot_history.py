import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_history(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("History JSON must be a non-empty list")
    return data


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from history.json")
    parser.add_argument('--history', type=Path, required=True, help='Path to history.json produced during training')
    parser.add_argument('--output', type=Path, default=None, help='Optional output image path (PNG)')
    args = parser.parse_args()

    history = load_history(args.history)
    epochs = [entry.get('epoch', idx) + 1 for idx, entry in enumerate(history)]
    epoch_time = [entry.get('epoch_time') for entry in history]
    train_loss = [entry.get('train_loss') for entry in history]

    metric_groups = {
        'Val Loss': [entry.get('val_loss') for entry in history],
        'Val Macro F1': [entry.get('val_macro_f1') for entry in history],
        'Val Micro F1': [entry.get('val_micro_f1') for entry in history],
        'Val AUROC': [entry.get('val_auroc') for entry in history],
        'Val AP': [entry.get('val_ap') for entry in history],
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    axes[0].plot(epochs, train_loss, marker='o', label='Train Loss')
    axes[0].set_title('Train Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')

    for ax, (name, values) in zip(axes[1:], metric_groups.items()):
        ax.plot(epochs, values, marker='o')
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.grid(True, linestyle='--', alpha=0.4)

    if len(axes) > len(metric_groups) + 1:
        ax_time = axes[len(metric_groups) + 1]
    else:
        fig_time, ax_time = plt.subplots(figsize=(6, 4))

    ax_time.plot(epochs, epoch_time, marker='o', color='tab:purple')
    ax_time.set_title('Epoch Time (s)'); ax_time.set_xlabel('Epoch'); ax_time.set_ylabel('Seconds')
    ax_time.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        if len(axes) <= len(metric_groups) + 1:
            fig_time.savefig(args.output.with_name(args.output.stem + '_time.png'), dpi=200)
        print(f"Saved plot(s) to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
