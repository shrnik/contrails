import pandas as pd
import numpy as np
import cv2
import os
import json
import glob
from tqdm import tqdm
from datetime import datetime

import utils.detection_utils as detection_utils


def load_gvccs_parquet(parquet_path):
    """
    Load a GVCCS parquet file and rename columns to match the pipeline's expected format.

    Parquet columns: FLIGHT_ID, CALL_SIGN, REGISTRATION, ICAO_TYPE, TIMESTAMP_S,
                     LONGITUDE, LATITUDE, ALTI_FT, GRND_SPD, AZIMUTH, ZENITH, PIXEL_X, PIXEL_Y

    Pipeline expects: ident, time, image_x, image_y, lat, lon, alt_gnss_meters
    """
    df = pd.read_parquet(parquet_path)
    df = df.rename(columns={
        'FLIGHT_ID': 'ident',
        'PIXEL_X': 'image_x',
        'PIXEL_Y': 'image_y',
        'LATITUDE': 'lat',
        'LONGITUDE': 'lon',
        'ALTI_FT': 'alt_ft',
    })
    df['time'] = pd.to_datetime(df['TIMESTAMP_S'], unit='s', utc=True)
    df['alt_gnss_meters'] = df['alt_ft'] * 0.3048
    df = df.dropna(subset=['image_x', 'image_y'])
    df = df.sort_values('time').reset_index(drop=True)
    return df


def load_gvccs_annotations(annotations_path):
    """Load COCO-format annotations and return images list and annotations indexed by image_id."""
    with open(annotations_path, 'r') as f:
        coco = json.load(f)

    images_by_id = {img['id']: img for img in coco['images']}

    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    return coco, images_by_id, annotations_by_image


def get_gvccs_image_df(images_dir, coco_images):
    """
    Build an image DataFrame from COCO image entries.

    Args:
        images_dir: Path to the images directory
        coco_images: List of COCO image dicts with 'file_name', 'id', 'time', 'video_id'

    Returns:
        DataFrame with columns: time, image_file, image_id, video_id
    """
    rows = []
    for img in coco_images:
        img_path = os.path.join(images_dir, img['file_name'])
        if not os.path.exists(img_path):
            continue
        # Image timestamps in GVCCS are local French time (Europe/Paris),
        # convert to UTC to match parquet TIMESTAMP_S (which is true UTC)
        local_time = pd.to_datetime(img['time']).tz_localize('Europe/Paris')
        rows.append({
            'time': local_time.tz_convert('UTC'),
            'image_file': img['file_name'],
            'image_id': img['id'],
            'video_id': img['video_id'],
        })
    image_df = pd.DataFrame(rows)
    image_df = image_df.sort_values('time').reset_index(drop=True)
    return image_df


def match_flights_to_image(df_flights, image_time, tolerance_s=15):
    """
    Find flight positions closest to the image timestamp within a tolerance window.
    Returns a DataFrame with one row per flight at the closest timestamp.
    """
    time_diff = (df_flights['time'] - image_time).abs()
    mask = time_diff <= pd.Timedelta(seconds=tolerance_s)
    df_nearby = df_flights[mask].copy()
    if df_nearby.empty:
        return df_nearby
    # Keep closest ping per flight
    df_nearby['_tdiff'] = (df_nearby['time'] - image_time).abs()
    df_closest = df_nearby.loc[df_nearby.groupby('ident')['_tdiff'].idxmin()]
    print(f"Matched {len(df_closest)} flights to image at {image_time} (tolerance {tolerance_s}s)")
    return df_closest.drop(columns=['_tdiff'])


def draw_gt_annotations(img, annotations, color=(0, 255, 0), thickness=1):
    """Draw ground-truth contrail segmentation polygons on the image."""
    for ann in annotations:
        for seg in ann['segmentation']:
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
            pts = pts.astype(np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            # Label with contrail type
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            label = ann.get('type', '')
            if ann.get('flight_id'):
                label += f" ({ann['flight_id'][:6]})"
            cv2.putText(img, label, (cx, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return img


def run_contrail_pipeline_gvccs(split='train', video_ids=None, max_videos=None, draw_gt=True,
                                detector='canny', yolo_model_path=None, yolo_conf=0.25):
    """
    Run the contrail detection pipeline on the GVCCS dataset.

    Args:
        split: 'train' or 'test'
        video_ids: Optional list of specific video IDs to process. If None, process all.
        max_videos: Optional limit on number of videos to process.
        draw_gt: Whether to overlay ground-truth annotations on the output video.
        detector: 'canny' for edge-based detection, 'yolo' for YOLO segmentation.
        yolo_model_path: Path to trained YOLO seg weights (required if detector='yolo').
        yolo_conf: YOLO confidence threshold (default: 0.25).
    """
    gvccs_dir = os.path.join(os.path.dirname(__file__), 'GVCCS')
    split_dir = os.path.join(gvccs_dir, split)
    images_dir = os.path.join(split_dir, 'images')
    parquet_dir = os.path.join(split_dir, 'parquet')
    annotations_path = os.path.join(split_dir, 'annotations.json')

    # Load annotations
    print("Loading annotations...")
    coco, images_by_id, annotations_by_image = load_gvccs_annotations(annotations_path)
    image_df = get_gvccs_image_df(images_dir, coco['images'])
    print(f"Loaded {len(image_df)} images, {len(coco['annotations'])} annotations")

    # Load all parquet files for flight data
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, '*.parquet')))
    print(f"Found {len(parquet_files)} parquet files")

    # Build a mapping: video_id -> parquet file (they correspond by order/time range)
    videos = sorted(coco['videos'], key=lambda v: v['id'])
    if video_ids is not None:
        videos = [v for v in videos if v['id'] in video_ids]
    if max_videos is not None:
        videos = videos[:max_videos]

    # Match parquet files to videos by time range
    video_parquet_map = {}
    for pf in parquet_files:
        fname = os.path.basename(pf).replace('.parquet', '')
        start_str, end_str = fname.split('_')
        pf_start = pd.to_datetime(start_str, format='%Y%m%d%H%M%S', utc=True)
        pf_end = pd.to_datetime(end_str, format='%Y%m%d%H%M%S', utc=True)
        for v in videos:
            v_start = pd.to_datetime(v['start']).tz_localize('Europe/Paris').tz_convert('UTC')
            v_end = pd.to_datetime(v['stop']).tz_localize('Europe/Paris').tz_convert('UTC')
            # Match if time ranges overlap
            if pf_start <= v_end and pf_end >= v_start:
                video_parquet_map[v['id']] = pf
                break

    # Load YOLO model if needed
    yolo_model = None
    if detector == 'yolo':
        from ultralytics import YOLO
        if yolo_model_path is None:
            raise ValueError("--yolo-model-path is required when using --detector yolo")
        print(f"Loading YOLO model from {yolo_model_path}...")
        yolo_model = YOLO(yolo_model_path)

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), f'gvccs_output_{split}_{detector}')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    # Per-aircraft-per-frame accuracy: for each matched flight on each frame,
    # check if GT has a "new" contrail annotation with that flight_id
    aircraft_stats = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    for video in tqdm(videos, desc="Processing videos"):
        vid = video['id']
        v_images = image_df[image_df['video_id'] == vid].reset_index(drop=True)
        if v_images.empty:
            continue

        # Load flight data for this video
        if vid not in video_parquet_map:
            print(f"No parquet file for video {vid}, skipping")
            continue
        df_flights = load_gvccs_parquet(video_parquet_map[vid])
        # Filter flights to only those within the video's time range
        # Video start/stop are in local French time (Europe/Paris), convert to UTC
        v_start = pd.to_datetime(video['start']).tz_localize('Europe/Paris').tz_convert('UTC')
        v_end = pd.to_datetime(video['stop']).tz_localize('Europe/Paris').tz_convert('UTC')
        df_flights = df_flights[(df_flights['time'] >= v_start) & (df_flights['time'] <= v_end)]
        print(f"\nVideo {vid}: {len(v_images)} images, {df_flights['ident'].nunique()} flights")

        # Setup video writer
        video_name = f"video_{vid}_{video['start'].replace(':', '-')}"
        output_video_path = os.path.join(output_dir, f'{video_name}.mp4')
        first_img = cv2.imread(os.path.join(images_dir, v_images.iloc[0]['image_file']))
        if first_img is None:
            continue
        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 5, (w, h))

        flights_with_contrails = []
        # Track per-flight: whether it was ever detected, and whether it ever had a GT contrail
        all_detected_idents = set()   # flights detected as contrail on any frame
        all_gt_idents = set()         # flights with GT "new" contrail on any frame
        all_visible_idents = set()    # flights visible in any frame

        for idx in range(len(v_images)):
            row = v_images.iloc[idx]
            curr_img_path = os.path.join(images_dir, row['image_file'])
            if idx > 0:
                prev_img_path = os.path.join(images_dir, v_images.iloc[idx - 1]['image_file'])
            else:
                prev_img_path = curr_img_path

            # Match flights to this image's timestamp
            df_filtered = match_flights_to_image(df_flights, row['time'], tolerance_s=15)

            # Run the core detection pipeline
            if detector == 'yolo':
                img_o, rectangles, edge_data, _ = detection_utils.process_image_with_yolo(
                    curr_img_path,
                    yolo_model=yolo_model,
                    timestamp=row['time'],
                    df_filtered=df_filtered,
                    df_upsampled=df_flights,
                    conf=yolo_conf,
                    angle_tolerance_deg=16,
                )
            else:
                img_o, rectangles, edge_data, _ = detection_utils.process_image_with_canny_edges(
                    curr_img_path,
                    prev_img_path=prev_img_path,
                    timestamp=row['time'],
                    df_filtered=df_filtered,
                    df_upsampled=df_flights,
                    min_line_length=40,
                    angle_tolerance_deg=16
                )
            # Ground-truth annotations for this frame
            gt_anns = annotations_by_image.get(row['image_id'], [])
            has_gt_contrails = len(gt_anns) > 0

            # GT flight_ids with "new" contrails on this frame
            gt_flight_ids = {ann['flight_id'] for ann in gt_anns
                             if ann.get('flight_id') and ann.get('type') == 'new'}

            # Track visible flights and GT for per-flight deduplication
            visible_in_frame = set(
                df_filtered[
                    df_filtered['image_x'].between(0, w) &
                    df_filtered['image_y'].between(0, h)
                ]['ident'].unique()
            )
            all_visible_idents |= visible_in_frame
            all_gt_idents |= (gt_flight_ids & visible_in_frame)

            if rectangles is None:
                video_writer.write(cv2.imread(curr_img_path))
                continue
            if draw_gt and gt_anns:
                img_o = draw_gt_annotations(img_o, gt_anns, color=(0, 255, 0), thickness=1)

            # Draw detection rectangles and detected lines
            for ident, (rect_poly, arrow, direction_info) in rectangles.items():
                edge_det = edge_data[ident]['is_making_contrails']

                # Color: green = contrail detected, blue = no detection
                color = (0, 255, 0) if edge_det else (255, 0, 0)

                cv2.polylines(img_o, [rect_poly], isClosed=True, color=color, thickness=2)

                # Show score
                x, y_coord, w_r, h_r = edge_data[ident]['bbox']
                cv2.putText(img_o, f"s={edge_data[ident]['score']:.2f}", (x, y_coord + h_r + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw detected Hough lines
                for x1, y1, x2, y2, length in edge_data[ident]['lines']:
                    cv2.line(img_o, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange lines

            # Record flights detected as making contrails
            for ident, (rect_poly, arrow, direction_info) in rectangles.items():
                edge_det = edge_data[ident]['is_making_contrails']
                if edge_det:
                    flight_row = df_filtered[df_filtered['ident'] == ident]
                    if not flight_row.empty:
                        result_row = flight_row.iloc[0].to_dict()
                        result_row['detection_time'] = row['time']
                        result_row['image_file'] = row['image_file']
                        result_row['image_id'] = row['image_id']
                        result_row['video_id'] = vid
                        result_row['gt_contrail_count'] = len(gt_anns)
                        result_row['score'] = edge_data[ident]['score']
                        result_row['edge_detection'] = edge_det
                        flights_with_contrails.append(result_row)

            # Track which flights were detected as making contrails
            for ident, (_, _, _) in rectangles.items():
                if edge_data[ident]['is_making_contrails']:
                    all_detected_idents.add(ident)

            # Draw flight positions on image
            for _, frow in df_filtered.iterrows():
                ix, iy = frow['image_x'], frow['image_y']
                if not np.isnan(ix) and not np.isnan(iy) and 0 <= ix < img_o.shape[1] and 0 <= iy < img_o.shape[0]:
                    cv2.circle(img_o, (int(ix), int(iy)), 5, (0, 0, 255), -1)
                    label = str(frow.get('CALL_SIGN', frow['ident'][:6]))
                    cv2.putText(img_o, label, (int(ix), int(iy) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            video_writer.write(img_o)

        video_writer.release()
        print(f"Video saved to {output_video_path}")

        # Compute per-flight accuracy for this video (each flight counted once)
        for ident in all_visible_idents:
            has_gt = ident in all_gt_idents
            was_detected = ident in all_detected_idents
            if has_gt and was_detected:
                aircraft_stats['TP'] += 1
            elif has_gt and not was_detected:
                aircraft_stats['FN'] += 1
            elif not has_gt and was_detected:
                aircraft_stats['FP'] += 1
            else:
                aircraft_stats['TN'] += 1

        # Save per-video results
        if flights_with_contrails:
            df_result = pd.DataFrame(flights_with_contrails)
            csv_path = os.path.join(output_dir, f'{video_name}_detections.csv')
            df_result.to_csv(csv_path, index=False)
            all_results.extend(flights_with_contrails)

    # Save combined results
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(os.path.join(output_dir, f'all_detections_{split}.csv'), index=False)
        print(f"\nTotal detections: {len(df_all)}")
        print(f"Unique flights detected making contrails: {df_all['ident'].nunique()}")

    # Print aircraft-level accuracy stats
    tp, fp, fn, tn = aircraft_stats['TP'], aircraft_stats['FP'], aircraft_stats['FN'], aircraft_stats['TN']
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"  Aircraft-Level Detection Stats ({split})")
    print(f"{'='*50}")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}  Total: {total}")
    print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  Accuracy: {accuracy:.3f}")
    print(f"{'='*50}")

    print(f"\nAll outputs saved to {output_dir}/")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run contrail detection pipeline on GVCCS dataset')
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--video-ids', type=int, nargs='+', default=None,
                        help='Specific video IDs to process')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Max number of videos to process')
    parser.add_argument('--no-gt', action='store_true',
                        help='Do not draw ground-truth annotations')
    parser.add_argument('--detector', default='canny', choices=['canny', 'yolo'],
                        help='Detection method: canny (edge-based) or yolo (segmentation)')
    parser.add_argument('--yolo-model-path', type=str, default=None,
                        help='Path to trained YOLO seg weights (required for --detector yolo)')
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                        help='YOLO confidence threshold (default: 0.25)')
    args = parser.parse_args()

    run_contrail_pipeline_gvccs(
        split=args.split,
        video_ids=args.video_ids,
        max_videos=args.max_videos,
        draw_gt=not args.no_gt,
        detector=args.detector,
        yolo_model_path=args.yolo_model_path,
        yolo_conf=args.yolo_conf,
    )


if __name__ == "__main__":
    main()
