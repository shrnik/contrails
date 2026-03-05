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


parser = argparse.ArgumentParser(description="Run contrail detection pipeline for Arizona camera.")
parser.add_argument("date", help="Date to process in YYYY-MM-DD format (e.g. 2026-01-19)")
args = parser.parse_args()
show_edges = True
store_contrail_rois = False

date_str = args.date
year, month, day = date_str.split('-')
adsb_csv_path = f"/Users/shrenikborad/pless/easy_adsb/data/arizona_{year}_{month}_{day}.csv"
camera_params_path = "/Users/shrenikborad/pless/groundcam-contrail-detection/calibration_data/uni_az/camera_params.json"
base_dir = f"/Users/shrenikborad/pless/contrails/arizona_images/{date_str}/cam2"
camera_name = "arizona"

# curr_date_df = df[df['time'] == date_str]

image_df = get_image_data_arizona(base_dir)

intrinsics, distortion, rvec, tvec, origin_gps = proj_utils.load_camera_parameters(camera_params_path)

start_time = pd.to_datetime(f"{date_str} 08:05:00").tz_localize('America/Phoenix').tz_convert('UTC')
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

# Define video parameters
output_path = f'output_video_{date_str}_{camera_name}_cleaned_background_removal_long.mp4'
img_def = cv2.imread(f"{base_dir}/{image_df.iloc[0]['image_file']}")
frame_height, frame_width = img_def.shape[0], img_def.shape[1]
fps = 10  # frames per second

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
# csv with time and ident of flights that made contrails
flights_with_contrails = []

toProcess = image_df
for idx, row in tqdm(toProcess.iterrows(), total=len(toProcess), desc="Processing images"):

    # img = cv2.imread(f"/Users/shrenikborad/Downloads/NNDL/images_uwisc/east/2025-10-01/east/{row['image_file']}")
    # if img is None:
    #     print(f"Could not read image {row['image_file']}")
    #     continue
    df_filtered = df_upsampled[df_upsampled['time'] == row['time']]
    curr_img_path = f"{base_dir}/{row['image_file']}"
    prev_img_path = None
    if idx > 0:
        prev_img_path = f"{base_dir}/{image_df.iloc[idx-1]['image_file']}"
    else:
        prev_img_path = curr_img_path
    img_o, rectangles, edge_data, edges_dict= detection_utils.process_image_with_canny_edges(f"{base_dir}/{row['image_file']}",
                                prev_img_path=prev_img_path,
                                timestamp=row['time'],
                                df_filtered=df_filtered,
                                df_upsampled=df_upsampled,
                                min_line_length=20)
    for ident, (rect_poly, arrow, direction_info) in rectangles.items():
        # Draw rectangle outline
        color = (255, 0, 0)  # Blue for normal
        if edge_data[ident]['is_making_contrails']:
            color = (0, 255, 255)  # Yellow for contrails
            
        cv2.polylines(img_o, [rect_poly], isClosed=True, color=color, thickness=2)
        if show_edges:
           edges_final = edge_data[ident]['edges']
           bbox = edge_data[ident]['bbox']
           x, y, w, h = bbox
           # Overlay edges on the combined image
           img_o[y:y+h, x:x+w][edges_final > 0] = (0, 255, 0)


        if edge_data[ident]['is_making_contrails']:
            row_to_append =  df_filtered[df_filtered['ident'] == ident]
            # save the cropped roi image of the contrail making aircraft
            x, y, w, h = edge_data[ident]['bbox']
            roi_img = img_o[y:y+h, x:x+w]
            if roi_img.size != 0 and store_contrail_rois:
                # show roi inline in plt
                roi_img_path = f"contrail_images/{date_str}/{camera_name}_contrail_{ident}_{row['time'].strftime('%H%M%S')}.jpg"
                # check if directory exists else create
                os.makedirs(os.path.dirname(roi_img_path), exist_ok=True)
                iswrite = cv2.imwrite(roi_img_path, roi_img)
                print(f"Written ROI image to {roi_img_path}: {iswrite}")
                # print(f"Saved contrail ROI image to {roi_img_path}")
                row_to_append = row_to_append.copy()
                row_to_append['contrail_image_path'] = roi_img_path
            # append the whole row with all the data
            flights_with_contrails.append(row_to_append)
            # for x1, y1, x2, y2, length in edge_data[ident]["lines"]:
            #     cv2.line(img_o, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange lines
            # if tip and base:
            #     cv2.arrowedLine(img_output, base, tip, (255, 255, 0), 2, tipLength=0.3)
    img = img_o
    for ident, image_x, image_y in zip(df_filtered['ident'], df_filtered['image_x'], df_filtered['image_y']):
        if not np.isnan(image_x) and not np.isnan(image_y) and 0 <= image_x < img.shape[1] and 0 <= image_y < img.shape[0]:
            cv2.circle(img, (int(image_x), int(image_y)), 5, (0, 0, 255), -1)
            cv2.putText(img, str(ident), (int(image_x), int(image_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    video_writer.write(img)
    print(f"Video saved to {output_path}")
    if len(flights_with_contrails) > 0:
        df_contrails = pd.concat(flights_with_contrails, ignore_index=True)
        df_contrails.to_csv(f'flights_with_contrails_{camera_name}_{date_str}.csv', index=False)
        print(f"CSV of flights with contrails saved to flights_with_contrails_{camera_name}_{date_str}.csv")



video_writer.release()
print(f"Video saved to {output_path}")