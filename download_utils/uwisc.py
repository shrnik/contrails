#!/usr/bin/env python3

import requests
from pathlib import Path
import re
from typing import List
import torch
from PIL import Image
import numpy as np
from urllib.parse import urljoin
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherCameraProcessor:
    def __init__(self, base_url: str = "https://metobs.ssec.wisc.edu/pub/cache/aoss/cameras/east/img/2025/01/09/orig/"):
        self.base_url = base_url
        self.images_dir = Path("downloaded_images/east")
        self.images_dir.mkdir(exist_ok=True)
        
        # # Initialize SigLIP-2 model for embeddings
        # self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-384")
        # self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-384")
        
        # # Initialize ChromaDB
        # self.chroma_client = chromadb.PersistentClient(path="./chroma-data")
        # try:
        #     self.collection = self.chroma_client.get_collection("east_camera")
        #     logger.info("Found existing ChromaDB collection")
        # except:
        #     self.collection = self.chroma_client.create_collection(
        #         name="east_camera",
        #         embedding_function=embedding_functions.DefaultEmbeddingFunction()
        #     )
        #     logger.info("Created new ChromaDB collection")

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

    def create_image_embeddings(self, image_paths: List[Path]) -> List[np.ndarray]:
        """Create embeddings for images using SigLIP-2 model."""
        embeddings = []
        
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            end = min(i + batch_size, len(image_paths))
            batch_paths = image_paths[i:end]
            batch_images = []
            valid_indices = []
            
            # Load all images in the batch
            for idx, img_path in enumerate(batch_paths):
                try:
                    image = Image.open(img_path).convert('RGB')
                    batch_images.append(image)
                    valid_indices.append(idx)
                    logger.info(f"Loaded image {img_path.name}")
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {e}")
                    # Add zero embedding placeholder for failed images
                    all_embeddings.append(np.zeros(768))
                
            if batch_images:
                try:
                # Process batch of images
                    inputs = self.processor(images=batch_images, return_tensors="pt")
                    
                    # Generate embeddings
                    with torch.no_grad():
                        image_features = self.model.get_image_features(**inputs)
                        # Normalize the embeddings
                        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    
                    # Add each embedding to the results
                    embeddings_numpy = image_features.numpy()
                    for idx, valid_idx in enumerate(valid_indices):
                        all_embeddings.append(embeddings_numpy[idx].flatten())
                        logger.info(f"Created embedding for {batch_paths[valid_idx].name}")
                
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # Add zero embeddings for the whole batch if batch processing fails
                    for _ in range(len(batch_images)):
                        all_embeddings.append(np.zeros(768))
            
        return all_embeddings

    def save_to_chromadb(self, image_paths: List[Path], embeddings: List[np.ndarray]):
        """Save image embeddings to ChromaDB."""
        for i, (img_path, embedding) in enumerate(zip(image_paths, embeddings)):
            try:
                doc_id = f"weather_img_{i}_{img_path.stem}"
                
                # Extract timestamp from filename for metadata
                timestamp_match = re.match(r'(\d{2})_(\d{2})_(\d{2})\.trig\+00', img_path.stem)
                if timestamp_match:
                    hour, minute, second = timestamp_match.groups()
                    timestamp = f"{hour}:{minute}:{second}"
                else:
                    timestamp = "unknown"
                
                metadata = {
                    "filename": img_path.name,
                    "timestamp": timestamp,
                    "url": urljoin(self.base_url, img_path.name),
                    "date": "2025-09-02"
                }
                
                # ChromaDB expects embeddings as lists
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[f"Weather camera image from {timestamp} on 2025-09-02"],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                
                logger.info(f"Saved {img_path.name} to ChromaDB")
                
            except Exception as e:
                logger.error(f"Error saving {img_path.name} to ChromaDB: {e}")

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
    processor = WeatherCameraProcessor()
    processor.download_images()


if __name__ == "__main__":
    main()