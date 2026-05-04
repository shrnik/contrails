#!/usr/bin/env python3

import re
import os
import argparse
import requests
from pathlib import Path
from typing import List
from urllib.parse import urljoin
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArizonaImagesDownloader:
    def __init__(self, base_url: str, output_dir: str):
        self.base_url = base_url
        os.makedirs(output_dir, exist_ok=True)
        self.images_dir = Path(output_dir)

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=30,
            pool_maxsize=30,
            max_retries=3,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_image_list(self) -> List[str]:
        try:
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            images = re.findall(r'href="(\d{14}\.jpg)"', response.text)
            images = sorted(set(images))
            logger.info(f"Found {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"Error fetching image list: {e}")
            return []

    def _download_single_image(self, img_name: str) -> Path:
        img_path = self.images_dir / img_name
        if img_path.exists():
            return img_path

        img_url = urljoin(self.base_url, img_name)
        try:
            response = self.session.get(img_url, timeout=60, stream=True)
            response.raise_for_status()
            with open(img_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            return img_path
        except Exception as e:
            logger.error(f"Error downloading {img_name}: {e}")
            return None

    def download_images(self, max_workers: int = 30) -> List[Path]:
        images = self.get_image_list()
        if not images:
            return []

        downloaded_paths = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._download_single_image, name): name for name in images}
            with tqdm(total=len(images), desc="Downloading images", unit="img") as pbar:
                for future in as_completed(futures):
                    path = future.result()
                    if path is not None:
                        downloaded_paths.append(path)
                    pbar.update(1)

        return downloaded_paths


def main():
    parser = argparse.ArgumentParser(description="Download JPG images from Arizona camera archive.")
    parser.add_argument("--date", "-d", required=True, help="Date in YYYY-MM-DD format (e.g. 2026-02-06)")
    parser.add_argument("--cam", "-c", default="cam2", help="Camera name (default: cam2)")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: arizona_images/<date>/<cam>)")
    parser.add_argument("--workers", "-w", type=int, default=30)
    args = parser.parse_args()

    date_parts = args.date.split("-")
    url = f"https://camera.cs.arizona.edu/archives/{date_parts[0]}/{date_parts[1]}/{date_parts[2]}/{args.cam}/"
    output_dir = args.output or f"arizona_images/{args.date}/{args.cam}"

    downloader = ArizonaImagesDownloader(base_url=url, output_dir=output_dir)
    paths = downloader.download_images(max_workers=args.workers)
    logger.info(f"Downloaded {len(paths)} images to {output_dir}")


if __name__ == "__main__":
    main()
