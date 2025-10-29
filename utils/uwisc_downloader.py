#!/usr/bin/env python3

import requests
from pathlib import Path
import re
from typing import List
from urllib.parse import urljoin
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagesDownloader:
    def __init__(self,date: str):
        path = f"downloaded_images/east/{date}/"
        os.makedirs(path, exist_ok=True)
        self.images_dir = Path(path)

    def get_image_list(self) -> List[str]:
        """Fetch the list of available images from the server."""
        try:
            response = requests.get(self.base_url, timeout=1000)
            response.raise_for_status()
            
            # Extract image filenames using regex
            image_pattern = r'(\d{2}_\d{2}_\d{2}\.trig\+00\.jpg)'
            images = re.findall(image_pattern, response.text)
            
            # Sort images to ensure consistent ordering
            images.sort()
            logger.info(f"Found {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"Error fetching image list: {e}")
            return []

    def _download_single_image(self, img_name: str) -> Path:
        """Download a single image."""
        img_url = urljoin(self.base_url, img_name)
        img_path = self.images_dir / img_name

        # Skip if already downloaded
        if img_path.exists():
            logger.info(f"Skipping {img_name} (already exists)")
            return img_path

        try:
            logger.info(f"Downloading {img_name}")
            response = requests.get(img_url, timeout=30)
            response.raise_for_status()

            with open(img_path, 'wb') as f:
                f.write(response.content)

            return img_path

        except Exception as e:
            logger.error(f"Error downloading {img_name}: {e}")
            return None

    def download_images(self, base_url: str, max_workers: int = 10) -> List[Path]:
        """Download every 6th image from the server concurrently."""
        images = self.get_image_list()
        # Log the total number of images found
        logger.info(f"Fetched {len(images)} images from server")

        # Make the list unique by converting to a set and back to a list
        images = sorted(list(set(images)))
        logger.info(f"After removing duplicates: {len(images)} unique images")
        if not images:
            return []

        # Select every 6th image
        selected_images = images
        logger.info(f"Selected {len(selected_images)} images (every 6th)")
        downloaded_paths = []

        # Download images concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_img = {executor.submit(self._download_single_image, img_name): img_name
                           for img_name in selected_images}

            for future in as_completed(future_to_img):
                img_path = future.result()
                if img_path is not None:
                    downloaded_paths.append(img_path)

        return downloaded_paths

def main():
    # download images for october
    BASE_URL = "https://metobs.ssec.wisc.edu/pub/cache/aoss/cameras/east/img/"
    year = 2025
    month = 10
    for day in range(1, 25):
        date_path = f"{year}/{month}/{day:02d}/orig/"
        processor = ImagesDownloader(date=f"{year}-{month:02d}-{day:02d}")
        processor.base_url = urljoin(BASE_URL, date_path)
        logger.info(f"Processing images for date: {year}-{month:02d}-{day:02d}")
        processor.download_images(processor.base_url)


if __name__ == "__main__":
    main()