#!/usr/bin/env python3

import requests
from pathlib import Path
import re
from typing import List
from urllib.parse import urljoin
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagesDownloader:
    def __init__(self, base_url: str = "https://metobs.ssec.wisc.edu/pub/cache/aoss/cameras/east/img/2025/01/09/orig/"):
        self.base_url = base_url
        self.images_dir = Path("downloaded_images/east")
        self.images_dir.mkdir(exist_ok=True)

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

    def download_images(self, max_workers: int = 10) -> List[Path]:
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

    def process_all(self):
        """Complete processing pipeline."""
        logger.info("Starting image processing pipeline...")
        
        # Step 1: Download images
        logger.info("Step 1: Downloading images...")
        image_paths = self.download_every_sixth_image()
        
        if not image_paths:
            logger.error("No images downloaded, stopping pipeline")
            return
        
        logger.info(f"Downloaded {len(image_paths)} images")
        
        # Step 2: Create embeddings
        logger.info("Step 2: Creating embeddings...")
        embeddings = self.create_image_embeddings(image_paths)
        
        # Step 3: Save to ChromaDB
        logger.info("Step 3: Saving to ChromaDB...")
        self.save_to_chromadb(image_paths, embeddings)
        
        logger.info("Pipeline completed successfully!")


def main():
    processor = ImagesDownloader()
    processor.download_images()


if __name__ == "__main__":
    main()