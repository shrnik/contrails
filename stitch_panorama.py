#!/usr/bin/env python3
"""
Stitch images from the 5 AOSS rooftop cameras into a panorama
and overlay ADS-B flight tracks.

Each camera's pixels are reprojected into azimuth/altitude space using
the calibration parameters, then composited onto a single equirectangular canvas.
ADS-B flight positions are converted to azimuth/altitude from the camera origin
and drawn on top.

Usage:
    python stitch_panorama.py --date 2025-01-09 --time 08:00
    python stitch_panorama.py --date 2025-01-09 --time 08:00 --adsb madison_pings_2025_01_09.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pymap3d as pm
import requests
from tqdm import tqdm

import utils.adsb_utils as adsb_utils

# Camera names and their calibration file paths
CAMERAS = {
    "east":      "uwisc/east/camera_params.json",
    "west":      "uwisc/west/camera_params.json",
    "north":     "uwisc/north/camera_params.json",
    "south":     "uwisc/south/camera_params.json",
    "northwest": "uwisc/northwest/camera_params.json",
}

BASE_URL = "https://metobs.ssec.wisc.edu/pub/cache/aoss/cameras"
IMG_W, IMG_H = 2592, 1944

# Panorama output settings
AZ_RANGE = (0, 360)       # degrees
ALT_RANGE = (-10, 80)     # degrees
PANO_SCALE = 10            # pixels per degree (used for polar radius)
POLAR_SIZE = 1800          # diameter of the polar canvas in pixels


def load_camera_params(path):
    """Load camera parameters from JSON."""
    with open(path) as f:
        p = json.load(f)
    K = np.array(p["intrinsics"], dtype=np.float64)
    dist = np.array(p["distortion"], dtype=np.float64).ravel()
    R = np.array(p["rotation"], dtype=np.float64)
    t = np.array(p["translation"], dtype=np.float64).reshape(3, 1)
    return K, dist, R, t


def pixel_to_az_alt(K, dist, R):
    """
    For every pixel in a 2592x1944 image, compute the azimuth and altitude
    it corresponds to in world coordinates (ENU).

    Returns:
        az_map: (H, W) array of azimuth in degrees [0, 360)
        alt_map: (H, W) array of altitude in degrees
    """
    # Create grid of all pixel coordinates
    u = np.arange(IMG_W, dtype=np.float32)
    v = np.arange(IMG_H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    # Reshape for OpenCV: (N, 1, 2)
    pts = np.stack([uu.ravel(), vv.ravel()], axis=-1).reshape(-1, 1, 2)

    # Undistort points to get normalized camera coordinates
    undist = cv2.undistortPoints(pts, K, dist)  # (N, 1, 2)
    x_norm = undist[:, 0, 0]
    y_norm = undist[:, 0, 1]

    # Ray directions in camera coords: (x_norm, y_norm, 1)
    rays_cam = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)  # (N, 3)

    # Transform to ENU world coords: R maps ENU->cam, so R.T maps cam->ENU
    rays_enu = (R.T @ rays_cam.T).T  # (N, 3)

    # ENU: East=x, North=y, Up=z
    east = rays_enu[:, 0]
    north = rays_enu[:, 1]
    up = rays_enu[:, 2]

    # Azimuth: angle from North, clockwise (geographic convention)
    az = np.degrees(np.arctan2(east, north)) % 360  # [0, 360)

    # Altitude: angle above horizon
    horiz = np.sqrt(east**2 + north**2)
    alt = np.degrees(np.arctan2(up, horiz))

    az_map = az.reshape(IMG_H, IMG_W)
    alt_map = alt.reshape(IMG_H, IMG_W)

    return az_map, alt_map


def list_available_images(camera_name, date_str, start_time, end_time):
    """
    List all available image filenames from the AOSS server for a camera/date
    within the given time range.

    Args:
        camera_name: e.g. "east"
        date_str: "YYYY-MM-DD"
        start_time: "HH:MM" local time (inclusive)
        end_time: "HH:MM" local time (exclusive)

    Returns:
        Sorted list of filenames like "08_00_08.trig+00.jpg"
    """
    import re
    year, month, day = date_str.split("-")
    dir_url = f"{BASE_URL}/{camera_name}/img/{year}/{month}/{day}/orig/"
    try:
        resp = requests.get(dir_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  WARNING: Could not list images for {camera_name}: {e}")
        return []

    pattern = r'(\d{2}_\d{2}_\d{2})\.trig\+00\.jpg'
    filenames = sorted(set(re.findall(pattern, resp.text)))

    # Filter to time range
    start_hhmm = start_time.replace(":", "_")
    end_hhmm = end_time.replace(":", "_")
    filtered = [f for f in filenames if start_hhmm <= f[:5] < end_hhmm]
    return [f"{f}.trig+00.jpg" for f in filtered]


def download_image_by_name(camera_name, date_str, filename):
    """Download a specific image by filename."""
    year, month, day = date_str.split("-")
    url = f"{BASE_URL}/{camera_name}/img/{year}/{month}/{day}/orig/{filename}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 1000:
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception:
        pass
    return None


def download_image(camera_name, date_str, time_str):
    """
    Download an image from the AOSS server.
    date_str: YYYY-MM-DD
    time_str: HH:MM or HH:MM:SS (local time CST/CDT)
    Returns the image as a numpy array, or None if not found.
    """
    year, month, day = date_str.split("-")
    parts = time_str.split(":")
    hh, mm = parts[0], parts[1]

    if len(parts) == 3:
        # Exact second specified
        ss = parts[2]
        filename = f"{hh}_{mm}_{ss}.trig+00.jpg"
        img = download_image_by_name(camera_name, date_str, filename)
        if img is not None:
            print(f"  Downloaded {camera_name}: {hh}_{mm}_{ss}")
            return img

    # Try multiple seconds offsets since cameras trigger at slightly different times
    for ss in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
        url = f"{BASE_URL}/{camera_name}/img/{year}/{month}/{day}/orig/{hh}_{mm}_{ss}.trig+00.jpg"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 1000:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    print(f"  Downloaded {camera_name}: {hh}_{mm}_{ss}")
                    return img
        except Exception:
            pass

    print(f"  WARNING: Could not find image for {camera_name} at {time_str}")
    return None


def az_alt_to_polar(az_deg, alt_deg):
    """
    Convert azimuth/altitude to polar canvas pixel coordinates.

    Polar layout:
      - Center of canvas = zenith (alt=90)
      - Edge of canvas = horizon (alt=ALT_RANGE[0])
      - North is up, East is right (clockwise azimuth)

    Returns:
        px, py: pixel coordinates on the POLAR_SIZE x POLAR_SIZE canvas
    """
    alt_min, alt_max = ALT_RANGE
    size = POLAR_SIZE
    center = size / 2.0
    max_radius = size / 2.0

    # Radius: horizon at edge, zenith at center
    # Normalize altitude: 0 at alt_max (zenith-ish), 1 at alt_min (horizon)
    r_norm = (alt_max - alt_deg) / (alt_max - alt_min)
    r_norm = np.clip(r_norm, 0, 1)
    radius = r_norm * max_radius

    # Angle: azimuth in radians, 0=North=up, clockwise
    # In image coords: up is -y, right is +x
    # angle from top, clockwise: x = r*sin(az), y = -r*cos(az)
    az_rad = np.radians(az_deg)
    px = center + radius * np.sin(az_rad)
    py = center - radius * np.cos(az_rad)

    return px, py


def build_panorama(images_dict, params_dict):
    """
    Build a circular polar panorama by projecting all camera images
    into azimuth/altitude space on a polar canvas.
    """
    size = POLAR_SIZE
    alt_min, alt_max = ALT_RANGE

    pano_sum = np.zeros((size, size, 3), dtype=np.float64)
    pano_weight = np.zeros((size, size), dtype=np.float64)

    for cam_name, img in images_dict.items():
        print(f"  Projecting {cam_name}...")
        K, dist, R, t = params_dict[cam_name]

        az_map, alt_map = pixel_to_az_alt(K, dist, R)

        # Weight map: higher weight toward image center
        cx, cy = IMG_W / 2, IMG_H / 2
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
        dist_from_center = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        weight = np.clip(1.0 - dist_from_center * 0.8, 0.05, 1.0)

        # Map to polar canvas
        pano_x, pano_y = az_alt_to_polar(az_map, alt_map)

        valid = (
            (pano_x >= 0) & (pano_x < size - 1) &
            (pano_y >= 0) & (pano_y < size - 1) &
            (alt_map > alt_min) & (alt_map < alt_max) &
            np.isfinite(pano_x) & np.isfinite(pano_y)
        )

        px = pano_x[valid].astype(np.int32)
        py = pano_y[valid].astype(np.int32)
        w = weight[valid]
        pixels = img[valid].astype(np.float64)

        np.add.at(pano_sum, (py, px), pixels * w[:, None])
        np.add.at(pano_weight, (py, px), w)

    mask = pano_weight > 0
    pano = np.zeros_like(pano_sum, dtype=np.uint8)
    pano[mask] = (pano_sum[mask] / pano_weight[mask, None]).astype(np.uint8)

    return pano, pano_weight


def add_grid_overlay(pano, az_range, alt_range, scale):
    """Add polar grid: concentric altitude rings and radial azimuth lines."""
    size = POLAR_SIZE
    alt_min, alt_max = alt_range
    center = size // 2
    max_radius = size // 2
    overlay = pano.copy()

    # Altitude rings every 10 degrees
    for alt in range(0, alt_max + 1, 10):
        r_norm = (alt_max - alt) / (alt_max - alt_min)
        r = int(r_norm * max_radius)
        if r > 0:
            cv2.circle(overlay, (center, center), r, (80, 80, 80), 1, cv2.LINE_AA)
            # Label the ring (place at the North spoke)
            label_y = center - r - 5
            if 10 <= label_y < size - 10:
                cv2.putText(overlay, f"{alt}", (center + 3, label_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # Azimuth spokes every 30 degrees
    cardinals = {0: "N", 30: "30", 60: "60", 90: "E", 120: "120", 150: "150",
                 180: "S", 210: "210", 240: "240", 270: "W", 300: "300", 330: "330"}
    for az in range(0, 360, 30):
        az_rad = np.radians(az)
        x_end = int(center + max_radius * np.sin(az_rad))
        y_end = int(center - max_radius * np.cos(az_rad))
        cv2.line(overlay, (center, center), (x_end, y_end), (80, 80, 80), 1, cv2.LINE_AA)

        # Label at the edge
        label_r = max_radius + 15
        lx = int(center + label_r * np.sin(az_rad))
        ly = int(center - label_r * np.cos(az_rad))
        label = cardinals.get(az, f"{az}")
        font_scale = 0.6 if label in ("N", "E", "S", "W") else 0.4
        # Center the text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(overlay, label, (lx - tw // 2, ly + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    return overlay


def gps_to_az_alt(lats, lons, alts, origin_gps):
    """
    Convert GPS positions to azimuth/altitude angles as seen from the camera origin.

    Args:
        lats, lons, alts: arrays of GPS coordinates
        origin_gps: [lat, lon, alt] of the camera

    Returns:
        az: azimuth in degrees [0, 360), measured clockwise from North
        alt: altitude in degrees above horizon
    """
    # Convert to ENU relative to camera origin
    east, north, up = pm.geodetic2enu(lats, lons, alts,
                                       origin_gps[0], origin_gps[1], origin_gps[2])

    # Azimuth: clockwise from North
    az = np.degrees(np.arctan2(east, north)) % 360

    # Altitude: angle above horizon
    horiz = np.sqrt(east**2 + north**2)
    alt_angle = np.degrees(np.arctan2(up, horiz))

    return az, alt_angle


def overlay_adsb(pano, df_flights, origin_gps, target_time_utc, az_range, alt_range,
                 time_window_sec=60):
    """
    Overlay ADS-B flight tracks onto the polar panorama.

    Draws flight positions within time_window_sec of target_time as dots,
    with trailing track lines from positions within a wider window.
    """
    alt_min, alt_max = alt_range
    size = POLAR_SIZE

    # Time windows
    t_start = target_time_utc - pd.Timedelta(seconds=time_window_sec)
    t_end = target_time_utc + pd.Timedelta(seconds=time_window_sec)
    trail_start = target_time_utc - pd.Timedelta(minutes=5)

    mask_trail = (df_flights['time'] >= trail_start) & (df_flights['time'] <= t_end)
    df_trail = df_flights[mask_trail].copy()

    if df_trail.empty:
        print("  No ADS-B data in time window")
        return pano

    az, alt_angle = gps_to_az_alt(
        df_trail['lat'].values,
        df_trail['lon'].values,
        df_trail['alt_gnss_meters'].values,
        origin_gps
    )

    df_trail['az'] = az
    df_trail['alt_angle'] = alt_angle

    # Convert to polar canvas coordinates
    px, py = az_alt_to_polar(df_trail['az'].values, df_trail['alt_angle'].values)
    df_trail['pano_x'] = px.astype(int)
    df_trail['pano_y'] = py.astype(int)

    # Filter to visible region (within canvas and altitude range)
    visible = (
        (df_trail['pano_x'] >= 0) & (df_trail['pano_x'] < size) &
        (df_trail['pano_y'] >= 0) & (df_trail['pano_y'] < size) &
        (df_trail['alt_angle'] > alt_min) & (df_trail['alt_angle'] < alt_max)
    )
    df_visible = df_trail[visible]

    if df_visible.empty:
        print("  No ADS-B tracks visible in panorama FOV")
        return pano

    # Assign colors per flight
    unique_idents = df_visible['ident'].unique()
    np.random.seed(42)
    colors = {}
    for ident in unique_idents:
        colors[ident] = tuple(int(c) for c in np.random.randint(100, 255, 3))

    print(f"  Overlaying {len(unique_idents)} flights in view")

    # Draw trails and current positions per flight
    for ident, group in df_visible.groupby('ident'):
        group = group.sort_values('time')
        color = colors[ident]
        pts = group[['pano_x', 'pano_y']].values

        # Draw trail line
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                # Skip if points are too far apart (wrapping or discontinuity)
                if abs(pts[i][0] - pts[i+1][0]) < size // 2 and abs(pts[i][1] - pts[i+1][1]) < size // 2:
                    cv2.line(pano, tuple(pts[i]), tuple(pts[i+1]), color, 1, cv2.LINE_AA)

        # Draw current position (within narrow time window) as a larger dot
        mask_current = (group['time'] >= t_start) & (group['time'] <= t_end)
        current = group[mask_current]
        if not current.empty:
            # Use the closest point to target time
            idx = (current['time'] - target_time_utc).abs().idxmin()
            cx, cy = int(current.loc[idx, 'pano_x']), int(current.loc[idx, 'pano_y'])
            cv2.circle(pano, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(pano, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

            # Label with flight ident
            label = str(ident)
            cv2.putText(pano, label, (cx + 7, cy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(pano, label, (cx + 7, cy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return pano


def precompute_projection_maps(params_dict):
    """
    Precompute polar projection maps and weight maps for each camera.
    This is expensive, so we do it once and reuse for every frame.
    """
    size = POLAR_SIZE
    alt_min, alt_max = ALT_RANGE
    maps = {}
    for cam_name, (K, dist, R, t) in params_dict.items():
        print(f"  Precomputing projection for {cam_name}...")
        az_map, alt_map = pixel_to_az_alt(K, dist, R)

        # Weight map: higher weight toward image center
        cx, cy = IMG_W / 2, IMG_H / 2
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
        dist_from_center = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        weight = np.clip(1.0 - dist_from_center * 0.8, 0.05, 1.0)

        pano_x, pano_y = az_alt_to_polar(az_map, alt_map)

        valid = (
            (pano_x >= 0) & (pano_x < size - 1) &
            (pano_y >= 0) & (pano_y < size - 1) &
            (alt_map > alt_min) & (alt_map < alt_max) &
            np.isfinite(pano_x) & np.isfinite(pano_y)
        )

        maps[cam_name] = {
            'pano_x': pano_x[valid].astype(np.int32),
            'pano_y': pano_y[valid].astype(np.int32),
            'weight': weight[valid],
            'valid_mask': valid,
        }
    return maps


def build_panorama_fast(images_dict, projection_maps):
    """
    Build a polar panorama using precomputed projection maps (much faster per frame).
    """
    size = POLAR_SIZE
    pano_sum = np.zeros((size, size, 3), dtype=np.float64)
    pano_weight = np.zeros((size, size), dtype=np.float64)

    for cam_name, img in images_dict.items():
        if cam_name not in projection_maps:
            continue
        m = projection_maps[cam_name]
        px, py, w = m['pano_x'], m['pano_y'], m['weight']
        pixels = img[m['valid_mask']].astype(np.float64)

        np.add.at(pano_sum, (py, px), pixels * w[:, None])
        np.add.at(pano_weight, (py, px), w)

    mask = pano_weight > 0
    pano = np.zeros_like(pano_sum, dtype=np.uint8)
    pano[mask] = (pano_sum[mask] / pano_weight[mask, None]).astype(np.uint8)

    return pano


def find_adsb_csv(script_dir, date_str, adsb_arg):
    """Locate ADS-B CSV file."""
    if adsb_arg:
        return adsb_arg
    year, month, day = date_str.split('-')
    candidates = [
        script_dir / f"adsb_flightpings_wisconsin_{date_str}.csv",
        Path(f"/Users/shrenikborad/pless/easy_adsb/data/madison_pings_{year}_{month}_{day}.csv"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def load_adsb_for_window(adsb_path, origin_gps, start_utc, end_utc, max_range_m=80000):
    """Load and upsample ADS-B data for a time window."""
    df = pd.read_csv(adsb_path)
    df['time'] = pd.to_datetime(df['time'])
    # Include extra padding for trails
    pad_start = start_utc - pd.Timedelta(minutes=10)
    pad_end = end_utc + pd.Timedelta(minutes=5)
    df = df[(df['time'] >= pad_start) & (df['time'] < pad_end)]
    if df.empty:
        return df
    df_upsampled = adsb_utils.get_upsampled_df_for_day(df, max_range_m=max_range_m)
    return df_upsampled


def main():
    parser = argparse.ArgumentParser(description="Stitch AOSS rooftop camera images into a panorama")
    parser.add_argument("--date", default="2025-01-09", help="Date (YYYY-MM-DD)")
    parser.add_argument("--time", default="08:00", help="Local time start (HH:MM)")
    parser.add_argument("--output", default=None, help="Output filename")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid overlay")
    parser.add_argument("--no-adsb", action="store_true", help="Disable ADS-B overlay")
    parser.add_argument("--adsb", default=None,
                        help="Path to ADS-B CSV file (auto-detected if not provided)")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Specific cameras to include (default: all)")
    parser.add_argument("--video", action="store_true", help="Generate a video instead of a single image")
    parser.add_argument("--duration", type=int, default=30,
                        help="Video duration in minutes (default: 10)")
    parser.add_argument("--fps", type=int, default=2,
                        help="Video frames per second (default: 2)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Select cameras
    cam_names = args.cameras if args.cameras else list(CAMERAS.keys())

    # Load calibration parameters
    print("Loading camera calibrations...")
    params = {}
    for name in cam_names:
        path = script_dir / CAMERAS[name]
        if not path.exists():
            print(f"  WARNING: calibration not found for {name}: {path}")
            continue
        params[name] = load_camera_params(str(path))
        print(f"  Loaded {name}")

    # Use the first camera's origin as the reference for ADS-B projection
    first_cam = list(params.keys())[0]
    cam_params_path = script_dir / CAMERAS[first_cam]
    with open(cam_params_path) as f:
        cp = json.load(f)
    origin_gps = [cp['origin_gps']['lat'], cp['origin_gps']['lon'], cp['origin_gps']['alt']]

    if args.video:
        _run_video(args, params, origin_gps, script_dir)
    else:
        _run_single(args, params, origin_gps, script_dir)


def _run_single(args, params, origin_gps, script_dir):
    """Generate a single panorama image."""
    # Download images
    print(f"\nDownloading images for {args.date} at {args.time}...")
    images = {}
    for name in params:
        img = download_image(name, args.date, args.time)
        if img is not None:
            images[name] = img

    if not images:
        print("ERROR: No images downloaded. Check date/time and network.")
        sys.exit(1)

    print(f"\nGot {len(images)} images: {list(images.keys())}")

    # Build panorama
    print("\nBuilding panorama...")
    pano, weight = build_panorama(images, params)

    # Add grid overlay
    if not args.no_grid:
        pano = add_grid_overlay(pano, AZ_RANGE, ALT_RANGE, PANO_SCALE)

    # Overlay ADS-B data
    if not args.no_adsb:
        adsb_path = find_adsb_csv(script_dir, args.date, args.adsb)
        if adsb_path and Path(adsb_path).exists():
            print(f"\nLoading ADS-B data from {adsb_path}...")
            target_local = pd.to_datetime(f"{args.date} {args.time}:00").tz_localize('America/Chicago')
            target_utc = target_local.tz_convert('UTC')
            df_upsampled = load_adsb_for_window(adsb_path, origin_gps, target_utc, target_utc)
            if not df_upsampled.empty:
                pano = overlay_adsb(pano, df_upsampled, origin_gps, target_utc,
                                    AZ_RANGE, ALT_RANGE)
        else:
            print("\nNo ADS-B CSV found. Use --adsb to specify path, or --no-adsb to skip.")

    # Save output
    out_path = args.output or f"panorama_{args.date}_{args.time.replace(':', '')}.jpg"
    cv2.imwrite(out_path, pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nSaved panorama to {out_path}")
    print(f"  Size: {pano.shape[1]}x{pano.shape[0]} px")


def _run_video(args, params, origin_gps, script_dir):
    """Generate a video of panorama frames with ADS-B overlay at native image cadence (~10s)."""
    start_hh, start_mm = map(int, args.time.split(':'))
    duration_min = args.duration
    fps = args.fps

    # Compute end time
    total_end = start_hh * 60 + start_mm + duration_min
    end_hh = total_end // 60
    end_mm = total_end % 60
    start_str = f"{start_hh:02d}:{start_mm:02d}"
    end_str = f"{end_hh:02d}:{end_mm:02d}"

    # List all available images from the first camera to get the native timestamps
    ref_cam = list(params.keys())[0]
    print(f"\nListing available images for {ref_cam} ({start_str} to {end_str})...")
    available = list_available_images(ref_cam, args.date, start_str, end_str)
    print(f"  Found {len(available)} images at native cadence (~10s intervals)")

    if not available:
        print("ERROR: No images found in time range.")
        sys.exit(1)

    # Extract HH:MM:SS timestamps from filenames like "08_00_08.trig+00.jpg"
    timestamps = []
    for fname in available:
        parts = fname.split(".")[0]  # "08_00_08"
        hh, mm, ss = parts.split("_")
        timestamps.append(f"{hh}:{mm}:{ss}")

    print(f"  {len(timestamps)} frames from {timestamps[0]} to {timestamps[-1]}")

    # Precompute projection maps (once, reused for all frames)
    print("\nPrecomputing projection maps...")
    proj_maps = precompute_projection_maps(params)

    # Polar canvas dimensions
    size = POLAR_SIZE

    # Load ADS-B data for the full window (once)
    df_adsb = None
    if not args.no_adsb:
        adsb_path = find_adsb_csv(script_dir, args.date, args.adsb)
        if adsb_path and Path(adsb_path).exists():
            print(f"\nLoading ADS-B data from {adsb_path}...")
            start_local = pd.to_datetime(f"{args.date} {start_str}:00").tz_localize('America/Chicago')
            end_local = pd.to_datetime(f"{args.date} {end_str}:00").tz_localize('America/Chicago')
            start_utc = start_local.tz_convert('UTC')
            end_utc = end_local.tz_convert('UTC')
            df_adsb = load_adsb_for_window(adsb_path, origin_gps, start_utc, end_utc)
            if df_adsb.empty:
                df_adsb = None
                print("  No ADS-B data in time window")
            else:
                print(f"  Loaded {df_adsb['ident'].nunique()} flights")
        else:
            print("\nNo ADS-B CSV found.")

    # Set up video writer — write raw to temp file, then re-encode with H.264 via ffmpeg
    out_path = args.output or f"panorama_{args.date}_{args.time.replace(':', '')}_{duration_min}min.mp4"
    tmp_path = out_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (size, size))

    # Generate each frame at native cadence
    frame_count = 0
    print(f"\nGenerating {len(timestamps)} frames...")
    for i, time_str in enumerate(tqdm(timestamps, desc="Frames")):
        hh, mm, ss = time_str.split(":")
        filename = f"{hh}_{mm}_{ss}.trig+00.jpg"

        # Download this exact image from all cameras
        images = {}
        for name in params:
            img = download_image_by_name(name, args.date, filename)
            if img is not None:
                images[name] = img

        if not images:
            continue

        # Build panorama using precomputed maps
        pano = build_panorama_fast(images, proj_maps)

        # Add grid
        if not args.no_grid:
            pano = add_grid_overlay(pano, AZ_RANGE, ALT_RANGE, PANO_SCALE)

        # Add timestamp label (bottom-left corner of polar canvas)
        label = f"{args.date} {hh}:{mm}:{ss} CST"
        cv2.putText(pano, label, (10, size - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pano, label, (10, size - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

        # Overlay ADS-B
        if df_adsb is not None:
            target_local = pd.to_datetime(
                f"{args.date} {hh}:{mm}:{ss}").tz_localize('America/Chicago')
            target_utc = target_local.tz_convert('UTC')
            pano = overlay_adsb(pano, df_adsb, origin_gps, target_utc,
                                AZ_RANGE, ALT_RANGE, time_window_sec=15)

        writer.write(pano)
        frame_count += 1

    writer.release()

    # Re-encode to H.264 for broad compatibility
    import subprocess
    print("\nRe-encoding to H.264...")
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        out_path
    ], capture_output=True)
    os.remove(tmp_path)

    print(f"\nSaved video to {out_path}")
    print(f"  {frame_count} frames, {fps} fps, ~{frame_count / fps:.0f}s playback")
    print(f"  Real time covered: {duration_min} min, native cadence: ~10s/frame")


if __name__ == "__main__":
    main()
