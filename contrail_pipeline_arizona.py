import math
import pandas as pd
import numpy as np
import cv2
import os
import argparse

import utils.adsb_utils as adsb_utils
import utils.projection_utils as proj_utils
from utils.image_data_utils import get_image_data_arizona, get_image_data_mit
import  utils.detection_utils as detection_utils
from tqdm import tqdm


def run_contrail_pipeline_arizona(date_str, base_dir=None, camera_params_path=None, adsb_csv_path=None,
                                   detector='canny', yolo_model_path=None, yolo_conf=0.25):
    show_edges = True
    store_contrail_rois = False
    camera_name = "arizona"

    year, month, day = date_str.split('-')
    base_dir = base_dir or f"/Users/shrenikborad/pless/contrails/arizona_images/{date_str}/cam2"
    camera_params_path = camera_params_path or "/Users/shrenikborad/pless/groundcam-contrail-detection/calibration_data/uni_az/camera_params.json"
    adsb_csv_path = adsb_csv_path or f"/Users/shrenikborad/pless/easy_adsb/data/arizona_{year}_{month}_{day}.csv"

    print(adsb_csv_path, "row count:", sum(1 for _ in open(adsb_csv_path)))
    image_df = get_image_data_arizona(base_dir)

    intrinsics, distortion, rvec, tvec, origin_gps = proj_utils.load_camera_parameters(camera_params_path)

    start_time = pd.to_datetime(f"{date_str} 08:00:00").tz_localize('America/Phoenix').tz_convert('UTC')
    end_time = pd.to_datetime(f"{date_str} 18:00:00").tz_localize('America/Phoenix').tz_convert('UTC')

    df = adsb_utils.read_adsblol_csv(adsb_csv_path, origin_gps=origin_gps)
    df = df[(df['time'] >= start_time) & (df['time'] < end_time)]
    df_upsampled = adsb_utils.get_upsampled_df_for_day(df, max_range_m=100000)

    image_x, image_y, cam_distance = proj_utils.gps_to_camxy_vasha_fixed(
        df_upsampled['lat'].values,
        df_upsampled['lon'].values,
        df_upsampled['alt_gnss_meters'].values,
        cam_k=intrinsics,
        cam_r=rvec,
        cam_t=tvec,
        camera_gps=origin_gps,
        distortion=distortion
    )

    df_upsampled['image_x'] = image_x
    df_upsampled['image_y'] = image_y
    df_upsampled['cam_distance'] = cam_distance
    image_df = image_df[(image_df['time'] >= start_time) & (image_df['time'] < end_time)]

    # Load YOLO model if needed
    yolo_model = None
    if detector == 'yolo':
        from ultralytics import YOLO
        if yolo_model_path is None:
            raise ValueError("--yolo-model-path is required when using --detector yolo")
        print(f"Loading YOLO model from {yolo_model_path}...")
        yolo_model = YOLO(yolo_model_path)

    # Define video parameters
    output_path = f'output_video_{date_str}_{camera_name}_cleaned_background_removal_long.mp4'
    img_def = cv2.imread(f"{base_dir}/{image_df.iloc[0]['image_file']}")
    frame_height, frame_width = img_def.shape[0], img_def.shape[1]
    fps = 10  # frames per second

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    flights_with_contrails = []

    toProcess = image_df.reset_index(drop=True)
    for idx, row in tqdm(toProcess.iterrows(), total=len(toProcess), desc="Processing images"):
        df_filtered = df_upsampled[df_upsampled['time'] == row['time']]
        curr_img_path = f"{base_dir}/{row['image_file']}"
        prev_img_path = None
        if idx > 0:
            prev_img_path = f"{base_dir}/{toProcess.iloc[idx-1]['image_file']}"
        else:
            prev_img_path = curr_img_path
        if detector == 'yolo':
            img_o, rectangles, edge_data, edges_dict = detection_utils.process_image_with_yolo(
                f"{base_dir}/{row['image_file']}",
                yolo_model=yolo_model,
                timestamp=row['time'],
                df_filtered=df_filtered,
                df_upsampled=df_upsampled,
                conf=yolo_conf,
                angle_tolerance_deg=16,
            )
        else:
            img_o, rectangles, edge_data, edges_dict = detection_utils.process_image_with_canny_edges(f"{base_dir}/{row['image_file']}",
                                    prev_img_path=prev_img_path,
                                    timestamp=row['time'],
                                    df_filtered=df_filtered,
                                    df_upsampled=df_upsampled,
                                    min_line_length=20)
        for ident, (rect_poly, arrow, direction_info) in rectangles.items():
            color = (255, 0, 0)  # Blue for normal
            if edge_data[ident]['is_making_contrails']:
                color = (0, 255, 255)  # Yellow for contrails

            cv2.polylines(img_o, [rect_poly], isClosed=True, color=color, thickness=2)
            if show_edges:
               edges_final = edge_data[ident]['edges']
               bbox = edge_data[ident]['bbox']
               x, y, w, h = bbox
               img_o[y:y+h, x:x+w][edges_final > 0] = (0, 255, 0)

            if edge_data[ident]['is_making_contrails']:
                row_to_append = df_filtered[df_filtered['ident'] == ident]
                x, y, w, h = edge_data[ident]['bbox']
                roi_img = img_o[y:y+h, x:x+w]
                flight_gps = row_to_append[['lat', 'lon', 'alt_gnss_meters']].values[0]
                # if roi_img.size != 0 and store_contrail_rois:
                #     roi_img_path = f"contrail_images/{date_str}/{camera_name}_contrail_{ident}_{row['time'].strftime('%H%M%S')}.jpg"
                #     os.makedirs(os.path.dirname(roi_img_path), exist_ok=True)
                #     iswrite = cv2.imwrite(roi_img_path, roi_img)
                #     print(f"Written ROI image to {roi_img_path}: {iswrite}")
                #     row_to_append = row_to_append.copy()
                #     row_to_append['contrail_image_path'] = roi_img_path

                lines = edge_data[ident]["lines"]
                longest_line = max(lines, key=lambda x: x[4]) if lines else None
                if longest_line:
                    # real world length
                    image_points = np.array([[longest_line[0], longest_line[1]], [longest_line[2], longest_line[3]]], dtype=np.float32)
                    flight_distance = detection_utils.get_flight_distance(flight_gps, origin_gps)
                    real_world_points = proj_utils.image_to_gps(image_points, k_matrix=intrinsics, dist_coeffs=distortion, r_matrix=rvec, t_vector=tvec, camera_gps=origin_gps, distance_m=flight_distance)
                    length_c = math.dist(real_world_points[0], real_world_points[1])
                    row_to_append["longest_contrail_length_meters"] = length_c
                flights_with_contrails.append(row_to_append)

        img = img_o
        for ident, image_x, image_y in zip(df_filtered['ident'], df_filtered['image_x'], df_filtered['image_y']):
            if not np.isnan(image_x) and not np.isnan(image_y) and 0 <= image_x < img.shape[1] and 0 <= image_y < img.shape[0]:
                cv2.circle(img, (int(image_x), int(image_y)), 5, (0, 0, 255), -1)
                cv2.putText(img, str(ident), (int(image_x), int(image_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        video_writer.write(img)

    video_writer.release()
    print(f"Video saved to {output_path}")
    if len(flights_with_contrails) > 0:
        df_contrails = pd.concat(flights_with_contrails, ignore_index=True)
        df_contrails.to_csv(f'flights_with_contrails_{camera_name}_{date_str}.csv', index=False)
        print(f"CSV of flights with contrails saved to flights_with_contrails_{camera_name}_{date_str}.csv")


def main():
    parser = argparse.ArgumentParser(description="Run contrail detection pipeline for Arizona camera.")
    parser.add_argument("dates", help="Date to process in YYYY-MM-DD format (e.g. 2026-01-19)")
    parser.add_argument("--base-dir", help="Base directory containing images (default: arizona_images/<date>/cam2)")
    parser.add_argument("--camera-params", dest="camera_params_path", help="Path to camera_params.json")
    parser.add_argument("--adsb-csv", dest="adsb_csv_path", help="Path to ADS-B CSV file")
    parser.add_argument("--detector", default="canny", choices=["canny", "yolo"],
                        help="Detection method: canny (edge-based) or yolo (segmentation)")
    parser.add_argument("--yolo-model-path", type=str, default=None,
                        help="Path to trained YOLO seg weights (required for --detector yolo)")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    args = parser.parse_args()

    for date_str in args.dates.split():
        run_contrail_pipeline_arizona(date_str, args.base_dir, args.camera_params_path, args.adsb_csv_path,
                                      detector=args.detector, yolo_model_path=args.yolo_model_path,
                                      yolo_conf=args.yolo_conf)

if __name__ == "__main__":
    main()
