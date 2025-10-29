import pandas as pd
import numpy as np
import cv2
import os

import utils.adsb_utils as adsb_utils
import utils.projection_utils as proj_utils
from utils.image_data_utils import get_image_data_uwisc
import  utils.detection_utils as detection_utils
from tqdm import tqdm


def run_contrail_pipeline_uwisc(date_str):

    adsb_csv_path = "./adsb_flightpings_MadisonWI_2025-10-01.csv"
    camera_params_path = "./uwisc/east/camera_params.json"
    base_dir = f'./downloaded_images/east/{date_str}'
    camera_name = "uwisc_east"

    df = pd.read_csv(adsb_csv_path)
    from_dt = pd.to_datetime(f"{date_str} 06:00:00").tz_localize('America/Chicago').tz_convert('UTC')
    to_dt = pd.to_datetime(f"{date_str} 19:00:00").tz_localize('America/Chicago').tz_convert('UTC')
    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['time'] >= from_dt) & (df['time'] < to_dt)]
    print(df.describe())
    df_upsampled = adsb_utils.get_upsampled_df_for_day(df, max_range_m=100000)


    # Load Camera Parameters
    intrinsics, distortion, rvec, tvec, origin_gps = proj_utils.load_camera_parameters(camera_params_path)

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


    image_df = get_image_data_uwisc(base_dir, date_str)
    image_df = image_df[(image_df['time'] >= from_dt) & (image_df['time'] < to_dt)]
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
        img_o, rectangles, edge_data= detection_utils.process_image_with_canny_edges(f"{base_dir}/{row['image_file']}",
                                    prev_img_path=prev_img_path,
                                    timestamp=row['time'],
                                    df_filtered=df_filtered,
                                    df_upsampled=df_upsampled)
        for ident, (rect_poly, arrow, direction_info) in rectangles.items():
            # Draw rectangle outline
            color = (255, 0, 0)  # Blue for normal
            if edge_data[ident]['is_making_contrails']:
                color = (0, 255, 255)  # Yellow for contrails
                
            cv2.polylines(img_o, [rect_poly], isClosed=True, color=color, thickness=2)
        

            if edge_data[ident]['is_making_contrails']:
                row_to_append =  df_filtered[df_filtered['ident'] == ident]
                # save the cropped roi image of the contrail making aircraft
                x, y, w, h = edge_data[ident]['bbox']
                roi_img = img_o[y:y+h, x:x+w]
                if roi_img.size != 0:
                    # show roi inline in plt
                    roi_img_path = f"contrail_images/{date_str}/{camera_name}_contrail_{ident}_{row['time'].strftime('%Y%m%d_%H%M%S')}.jpg"
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
            # Draw arrow if available
            if arrow:
                tip, base = arrow
                # if tip and base:
                #     cv2.arrowedLine(img_output, base, tip, (255, 255, 0), 2, tipLength=0.3)
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
    for day in range(1, 10):
        date_str = f"2025-10-{day:02d}"
        run_contrail_pipeline_uwisc(date_str)