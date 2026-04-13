#!/bin/bash
# Submit one SLURM job per (date, detector) combination.
#
# Usage:
#   ./submit_jobs.sh                    # submit all dates with both canny & yolo
#   ./submit_jobs.sh canny              # submit all dates with canny only
#   ./submit_jobs.sh yolo               # submit all dates with yolo only

set -euo pipefail

DETECTORS=${1:-"canny yolo"}
YOLO_MODEL=/SEAS/home/g32121899/visor/runs/segment/runs/gvccs_seg/yolo11n_seg/weights/best.pt
YOLO_CONF=0.25
OUTPUT_DIR="batch_results"
LABELS_DIR="labels"
LOGS_DIR="logs"

mkdir -p "$LOGS_DIR" "$OUTPUT_DIR"

for DETECTOR in $DETECTORS; do

    if [ "$DETECTOR" == "yolo" ]; then
        DETECTOR_ARGS="--detector yolo --yolo-model-path $YOLO_MODEL --yolo-conf $YOLO_CONF"
    else
        DETECTOR_ARGS="--detector canny"
    fi

    # ---- Arizona dates ----
    for f in "$LABELS_DIR"/arizona/*.csv; do
        [ -f "$f" ] || continue
        DATE=$(basename "$f" .csv)
        JOB_NAME="contrail_arizona_${DATE}_${DETECTOR}"

        sbatch \
            --job-name="$JOB_NAME" \
            --output="$LOGS_DIR/${JOB_NAME}_%j.log" \
            --error="$LOGS_DIR/${JOB_NAME}_%j.log" \
            -N 1 \
            -p large-gpu \
            --cpus-per-task=16 \
            --mem=128G \
            --mail-user=s.borad@gwu.edu \
            --mail-type=ALL \
            -t 12:00:00 \
            --wrap="source visor/bin/activate && cd contrails && python contrail_pipeline_arizona.py $DATE $DETECTOR_ARGS && mkdir -p $OUTPUT_DIR && mv flights_with_contrails_arizona_${DATE}*.csv $OUTPUT_DIR/ 2>/dev/null || true"

        echo "Submitted: $JOB_NAME"
    done

    # ---- UWisc dates ----
    for f in "$LABELS_DIR"/uwisc/*.csv; do
        [ -f "$f" ] || continue
        DATE=$(basename "$f" .csv)
        JOB_NAME="contrail_uwisc_${DATE}_${DETECTOR}"

        sbatch \
            --job-name="$JOB_NAME" \
            --output="$LOGS_DIR/${JOB_NAME}_%j.log" \
            --error="$LOGS_DIR/${JOB_NAME}_%j.log" \
            -N 1 \
            -p large-gpu \
            --cpus-per-task=16 \
            --mem=128G \
            --mail-user=s.borad@gwu.edu \
            --mail-type=ALL \
            -t 12:00:00 \
            --wrap="source visor/bin/activate && cd contrails && python contrail_pipeline_uwisc.py $DATE --camera-side east $DETECTOR_ARGS && mkdir -p $OUTPUT_DIR && mv flights_with_contrails_uwisc_east_${DATE}*.csv $OUTPUT_DIR/ 2>/dev/null || true"

        echo "Submitted: $JOB_NAME"
    done

done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
