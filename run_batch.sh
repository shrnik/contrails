#!/bin/bash
#SBATCH --job-name=contrail_batch
#SBATCH --output=logs/contrail_batch_%j.log
#SBATCH --error=logs/contrail_batch_%j.log
#SBATCH -N 1
#SBATCH -p med-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-user=s.borad@gwu.edu
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00

# Batch script to run contrail detection on all dates with labels
# (both Arizona and UWisc), then plot accuracy.
#
# Usage (local):
#   ./run_batch.sh [--detector canny|yolo] [--output-dir DIR]
#
# Usage (SLURM):
#   sbatch run_batch.sh

set -euo pipefail

mkdir -p logs
source visor/bin/activate
cd contrails

DETECTOR=${1:-canny}
YOLO_MODEL=/SEAS/home/g32121899/visor/runs/segment/runs/gvccs_seg/yolo11n_seg/weights/best.pt
YOLO_CONF=0.25
OUTPUT_DIR="batch_results"
LABELS_DIR="labels"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Contrail Detection Batch Run"
echo "  Detector: $DETECTOR"
echo "  Output:   $OUTPUT_DIR"
echo "============================================"

# Build detector args
if [ "$DETECTOR" == "yolo" ]; then
    DETECTOR_ARGS="--detector yolo --yolo-model-path $YOLO_MODEL --yolo-conf $YOLO_CONF"
else
    DETECTOR_ARGS="--detector canny"
fi

# ---- Arizona dates (from labels/arizona/*.csv) ----
for f in "$LABELS_DIR"/arizona/*.csv; do
    [ -f "$f" ] || continue
    DATE=$(basename "$f" .csv)
    echo ""
    echo ">> Arizona $DATE ..."
    python contrail_pipeline_arizona.py "$DATE" $DETECTOR_ARGS || {
        echo "WARNING: Arizona $DATE failed, skipping."
        continue
    }
    for csv in flights_with_contrails_arizona_"$DATE"*.csv; do
        [ -f "$csv" ] && mv "$csv" "$OUTPUT_DIR/"
    done
done

# ---- UWisc dates (from labels/uwisc/*.csv) ----
for f in "$LABELS_DIR"/uwisc/*.csv; do
    [ -f "$f" ] || continue
    DATE=$(basename "$f" .csv)
    echo ""
    echo ">> UWisc east $DATE ..."
    python contrail_pipeline_uwisc.py "$DATE" --camera-side east $DETECTOR_ARGS || {
        echo "WARNING: UWisc $DATE failed, skipping."
        continue
    }
    for csv in flights_with_contrails_uwisc_east_"$DATE"*.csv; do
        [ -f "$csv" ] && mv "$csv" "$OUTPUT_DIR/"
    done
done

# ---- Plot accuracy ----
echo ""
echo ">> Running accuracy plots ..."
python plot_detection_accuracy.py \
    --labels-dir "$LABELS_DIR" \
    --predictions-dir "$OUTPUT_DIR" \
    --save-dir "$OUTPUT_DIR/plots" \
    --no-show

echo ""
echo "Done. Results in $OUTPUT_DIR/, plots in $OUTPUT_DIR/plots/"
