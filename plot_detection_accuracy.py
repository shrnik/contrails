"""
Plot contrail detection accuracy: scatter plot and heatmap.

Usage:
    # Single run:
    python plot_detection_accuracy.py \
        --labels labels/uwisc/2025-01-09.csv \
        --predictions final_data2/flights_with_contrails_uwisc_east_2025-01-09.csv \
        --date 2025-01-09 \
        [--save-dir output/] [--bin-size 30min] [--detection-multiplier 5]

    # Batch run (all files in predictions dir matched to labels dir):
    python plot_detection_accuracy.py \
        --labels-dir labels/ \
        --predictions-dir final_data2/ \
        [--save-dir output/] [--bin-size 30min] [--detection-multiplier 5]
"""

import argparse
import glob
import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


CONTRAIL_POSITIVE_LABELS = {'Persistent', 'Dissipate < 10', 'Dissipate > 10'}
EXCLUDED_LABELS = {'Blocked'}


def load_and_merge(labels_path: str, predictions_path: str, date_str: str) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path)
    detections_df = pd.read_csv(predictions_path)
    detected_idents = set(detections_df['ident'].unique())

    eval_df = labels_df[~labels_df['label'].isin(EXCLUDED_LABELS)].copy()
    eval_df['ground_truth'] = eval_df['label'].isin(CONTRAIL_POSITIVE_LABELS).astype(int)
    eval_df['predicted'] = eval_df['ident'].isin(detected_idents).astype(int)

    eval_df['result'] = 'TN'
    eval_df.loc[(eval_df['ground_truth'] == 1) & (eval_df['predicted'] == 1), 'result'] = 'TP'
    eval_df.loc[(eval_df['ground_truth'] == 1) & (eval_df['predicted'] == 0), 'result'] = 'FN'
    eval_df.loc[(eval_df['ground_truth'] == 0) & (eval_df['predicted'] == 1), 'result'] = 'FP'

    eval_df['time_dt'] = pd.to_datetime(date_str + ' ' + eval_df['time_appear'])
    eval_df['avg_fl'] = ((eval_df['alt_min_ft'] + eval_df['alt_max_ft']) / 200).round(0).astype(int)
    return eval_df


def plot_scatter(df: pd.DataFrame, date_str: str, save_path: str = None, show: bool = True):
    result_styles = {
        'TP': {'color': '#2ecc71', 'marker': 'o', 'label': 'TP (contrail, detected)'},
        'FN': {'color': '#e74c3c', 'marker': 'X', 'label': 'FN (contrail, missed)'},
        'TN': {'color': '#3498db', 'marker': 's', 'label': 'TN (clear, not detected)'},
        'FP': {'color': '#f39c12', 'marker': '^', 'label': 'FP (clear, falsely detected)'},
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for result, style in result_styles.items():
        subset = df[df['result'] == result]
        ax.scatter(subset['time_dt'], subset['avg_fl'],
                   c=style['color'], marker=style['marker'],
                   s=80, alpha=0.85, label=style['label'], zorder=3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    fig.autofmt_xdate()
    ax.set_xlabel('Time of Day (UTC)')
    ax.set_ylabel('Flight Level')
    ax.set_title(f'Flight Level vs Time of Day — Contrail Detection Results ({date_str})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
    if show:
        plt.show()
    return fig


def plot_heatmap(df: pd.DataFrame, date_str: str, bin_size: str = '30min',
                 detection_multiplier: int = 1, save_path: str = None, show: bool = True):
    fl_min = (df['avg_fl'].min() // 10) * 10
    fl_max = ((df['avg_fl'].max() // 10) + 1) * 10
    fl_bins = np.arange(fl_min, fl_max + 10, 10)
    df = df.copy()
    df['fl_bin'] = pd.cut(df['avg_fl'], bins=fl_bins, labels=fl_bins[:-1])

    min_time, max_time = df['time_dt'].min(), df['time_dt'].max()
    time_bins = pd.date_range(start=min_time.floor(bin_size), end=max_time.ceil(bin_size), freq=bin_size)
    df['time_bin'] = pd.cut(df['time_dt'], bins=time_bins, labels=time_bins[:-1])

    def aggregate(group, col):
        return group[col].sum() * detection_multiplier - (1 - group[col]).sum()

    def make_heatmap(label_col):
        pivot = df.groupby(['time_bin', 'fl_bin'], observed=True).apply(
            lambda g: aggregate(g, label_col), include_groups=False
        ).reset_index().rename(columns={0: 'score'})
        return pivot.pivot(index='fl_bin', columns='time_bin', values='score').fillna(0)

    gt_heatmap = make_heatmap('ground_truth')
    pred_heatmap = make_heatmap('predicted')

    all_fl = sorted(set(gt_heatmap.index) | set(pred_heatmap.index))
    all_times = sorted(set(gt_heatmap.columns) | set(pred_heatmap.columns))
    gt_heatmap = gt_heatmap.reindex(index=all_fl, columns=all_times, fill_value=0).sort_index(ascending=False)
    pred_heatmap = pred_heatmap.reindex(index=all_fl, columns=all_times, fill_value=0).sort_index(ascending=False)

    vmax = max(abs(gt_heatmap.values.min()), abs(gt_heatmap.values.max()),
               abs(pred_heatmap.values.min()), abs(pred_heatmap.values.max()))
    vmin = -vmax

    mins_per_bin = pd.tseries.frequencies.to_offset(bin_size).nanos / 6e10
    ticks_per_hour = max(1, int(60 / mins_per_bin))
    n_time_bins = len(gt_heatmap.columns)
    fig_width = min(32, max(16, n_time_bins * 0.5))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 10))

    for ax, heatmap, title in [
        (ax1, gt_heatmap, f'Ground Truth Contrails by Flight Level and Time: {date_str}'),
        (ax2, pred_heatmap, f'Predicted Contrails by Flight Level and Time: {date_str}'),
    ]:
        im = ax.imshow(heatmap.values, aspect='auto', cmap='PuOr_r', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Flight Level (100s of feet)', fontsize=12)

        y_ticks = np.arange(len(heatmap.index))
        ax.set_yticks(y_ticks[::2])
        ax.set_yticklabels([f'FL{int(fl)}' for fl in heatmap.index[::2]])

        x_ticks = np.arange(len(heatmap.columns))
        ax.set_xticks(x_ticks[::ticks_per_hour])
        ax.set_xticklabels([t.strftime('%H:%M') for t in heatmap.columns[::ticks_per_hour]], rotation=45)
        plt.colorbar(im, ax=ax, pad=0.02).set_label('Score (Orange=Contrail, Purple=Clear)', fontsize=10)

    ax2.set_xlabel(f'Time ({bin_size} bins)', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(df: pd.DataFrame, date_str: str, save_path: str = None, show: bool = True):
    counts = {r: (df['result'] == r).sum() for r in ('TP', 'FP', 'FN', 'TN')}
    matrix = np.array([[counts['TP'], counts['FN']],
                        [counts['FP'], counts['TN']]])

    total = matrix.sum()
    precision = counts['TP'] / (counts['TP'] + counts['FP']) if (counts['TP'] + counts['FP']) > 0 else 0
    recall    = counts['TP'] / (counts['TP'] + counts['FN']) if (counts['TP'] + counts['FN']) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (counts['TP'] + counts['TN']) / total if total > 0 else 0

    print(f"Confusion Matrix:\n{matrix}")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, Accuracy: {accuracy:.2f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nPositive', 'Predicted\nNegative'], fontsize=11)
    ax.set_yticklabels(['Actual\nPositive', 'Actual\nNegative'], fontsize=11)

    for i in range(2):
        for j in range(2):
            label = ['TP', 'FN', 'FP', 'TN'][i * 2 + j]
            ax.text(j, i, f'{label}\n{matrix[i, j]}',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if matrix[i, j] > matrix.max() * 0.5 else 'black')

    ax.set_title(f'Confusion Matrix — {date_str}', fontsize=13, fontweight='bold', pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    metrics_text = f'Precision: {precision:.2f}   Recall: {recall:.2f}   F1: {f1:.2f}   Accuracy: {accuracy:.2f}'
    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=10, color='#333333')

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    if show:
        plt.show()
    return fig


def compute_stats(df: pd.DataFrame) -> dict:
    counts = {r: int((df['result'] == r).sum()) for r in ('TP', 'FP', 'FN', 'TN')}
    total = sum(counts.values())
    precision = counts['TP'] / (counts['TP'] + counts['FP']) if (counts['TP'] + counts['FP']) > 0 else 0
    recall = counts['TP'] / (counts['TP'] + counts['FN']) if (counts['TP'] + counts['FN']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (counts['TP'] + counts['TN']) / total if total > 0 else 0
    return {**counts, 'total': total, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}


def _compute_metrics(agg: dict) -> dict:
    prec = agg['TP'] / (agg['TP'] + agg['FP']) if (agg['TP'] + agg['FP']) > 0 else 0
    rec = agg['TP'] / (agg['TP'] + agg['FN']) if (agg['TP'] + agg['FN']) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc = (agg['TP'] + agg['TN']) / agg['total'] if agg['total'] > 0 else 0
    return {'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc}


def _aggregate(stats_list: list[dict]) -> dict:
    return {k: sum(s[k] for s in stats_list) for k in ('TP', 'FP', 'FN', 'TN', 'total')}


def _latex_row(date, source, camera, agg, bold=False):
    m = _compute_metrics(agg)
    vals = [date, source, camera,
            str(agg['TP']), str(agg['FP']), str(agg['FN']), str(agg['TN']), str(agg['total']),
            f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}", f"{m['accuracy']:.2f}"]
    if bold:
        vals = [f"\\textbf{{{v}}}" for v in vals]
    return '        ' + ' & '.join(vals) + r' \\'


def print_summary_table(all_stats: list[dict]):
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'    \centering')
    lines.append(r'    \caption{Contrail Detection Accuracy Summary}')
    lines.append(r'    \label{tab:detection_accuracy}')
    lines.append(r'    \begin{tabular}{lll rrrr r rrrr}')
    lines.append(r'        \toprule')
    lines.append(r'        Date & Source & Camera & TP & FP & FN & TN & Total & Prec & Rec & F1 & Acc \\')
    lines.append(r'        \midrule')

    # Per-day rows
    for s in all_stats:
        camera = s.get('camera', 'all')
        lines.append(_latex_row(s['date'], s['source'], camera, s))

    # Per-camera aggregates
    cameras = sorted(set(s.get('camera', 'all') for s in all_stats))
    if len(cameras) > 1 or (len(cameras) == 1 and cameras[0] != 'all'):
        lines.append(r'        \midrule')
        for cam in cameras:
            cam_stats = [s for s in all_stats if s.get('camera', 'all') == cam]
            agg = _aggregate(cam_stats)
            lines.append(_latex_row('All Dates', 'all', cam, agg, bold=True))

    # Overall total
    lines.append(r'        \midrule')
    agg = _aggregate(all_stats)
    lines.append(_latex_row('Total', 'all', 'all', agg, bold=True))

    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'\end{table}')

    latex_str = '\n'.join(lines)
    print(f"\n{latex_str}\n")


def find_pairs(labels_dir: str, predictions_dir: str) -> list[dict]:
    """Match prediction CSVs to label CSVs by date."""
    pairs = []
    for pred_file in sorted(glob.glob(os.path.join(predictions_dir, '*.csv'))):
        fname = os.path.basename(pred_file)
        # Extract date (YYYY-MM-DD) from filename
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
        if not date_match:
            continue
        date_str = date_match.group(1)

        # Determine source and camera from filename
        # e.g. flights_with_contrails_uwisc_east_2025-01-09.csv -> source=uwisc, camera=east
        # e.g. flights_with_contrails_arizona_2026-01-18.csv -> source=arizona, camera=all
        if 'arizona' in fname:
            source = 'arizona'
            camera = 'all'
        elif 'uwisc' in fname:
            source = 'uwisc'
            cam_match = re.search(r'uwisc_(\w+)_\d{4}', fname)
            camera = cam_match.group(1) if cam_match else 'all'
        else:
            source = 'unknown'
            camera = 'all'

        # Look for matching label file in subdirs
        label_path = os.path.join(labels_dir, source, f'{date_str}.csv')
        if not os.path.exists(label_path):
            # Try flat structure
            label_path = os.path.join(labels_dir, f'{date_str}.csv')
        if not os.path.exists(label_path):
            print(f"WARNING: No label file found for {fname} (tried {label_path}), skipping.")
            continue

        pairs.append({'labels': label_path, 'predictions': pred_file, 'date': date_str, 'source': source, 'camera': camera})
    return pairs


def run_single(labels_path, predictions_path, date_str, source, save_dir, bin_size, detection_multiplier, show, camera='all'):
    tag = f"{source}_{camera}_{date_str}" if camera != 'all' else f"{source}_{date_str}"
    print(f"\n{'='*60}")
    print(f"  Processing: {tag}")
    print(f"  Labels:      {labels_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"{'='*60}")

    df = load_and_merge(labels_path, predictions_path, date_str)
    stats = compute_stats(df)
    stats['date'] = date_str
    stats['source'] = source
    stats['camera'] = camera

    scatter_save = f"{save_dir}/scatter_{tag}.png" if save_dir else None
    heatmap_save = f"{save_dir}/heatmap_{tag}.png" if save_dir else None
    confmat_save = f"{save_dir}/confmat_{tag}.png" if save_dir else None

    plot_scatter(df, f"{source} {date_str}", save_path=scatter_save, show=show)
    plot_heatmap(df, f"{source} {date_str}", bin_size=bin_size,
                 detection_multiplier=detection_multiplier, save_path=heatmap_save, show=show)
    plot_confusion_matrix(df, f"{source} {date_str}", save_path=confmat_save, show=show)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Plot contrail detection accuracy')
    # Single-run args
    parser.add_argument('--labels', default=None, help='Path to ground truth labels CSV (single run)')
    parser.add_argument('--predictions', default=None, help='Path to detections CSV (single run)')
    parser.add_argument('--date', default=None, help='Date string YYYY-MM-DD (single run)')
    # Batch-run args
    parser.add_argument('--labels-dir', default=None, help='Directory containing label CSVs (batch run)')
    parser.add_argument('--predictions-dir', default=None, help='Directory containing prediction CSVs (batch run)')
    # Common args
    parser.add_argument('--save-dir', default=None, help='Directory to save plots')
    parser.add_argument('--bin-size', default='30min', help='Time bin size for heatmap (default: 30min)')
    parser.add_argument('--detection-multiplier', type=int, default=5,
                        help='Score multiplier for contrail detections (default: 5)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots (useful for batch)')
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    show = not args.no_show

    # Batch mode
    if args.labels_dir and args.predictions_dir:
        pairs = find_pairs(args.labels_dir, args.predictions_dir)
        if not pairs:
            print("ERROR: No matching label/prediction pairs found.")
            return
        all_stats = []
        for pair in pairs:
            stats = run_single(pair['labels'], pair['predictions'], pair['date'],
                               pair['source'], args.save_dir, args.bin_size,
                               args.detection_multiplier, show, camera=pair['camera'])
            all_stats.append(stats)
        print_summary_table(all_stats)

    # Single mode
    elif args.labels and args.predictions and args.date:
        fname = os.path.basename(args.predictions)
        source = 'arizona' if 'arizona' in fname else 'uwisc'
        cam_match = re.search(r'uwisc_(\w+)_\d{4}', fname)
        camera = cam_match.group(1) if cam_match else 'all'
        stats = run_single(args.labels, args.predictions, args.date,
                           source, args.save_dir, args.bin_size,
                           args.detection_multiplier, show, camera=camera)
        print_summary_table([stats])

    else:
        parser.error('Provide either --labels-dir and --predictions-dir (batch), '
                     'or --labels, --predictions, and --date (single run).')


if __name__ == '__main__':
    main()
