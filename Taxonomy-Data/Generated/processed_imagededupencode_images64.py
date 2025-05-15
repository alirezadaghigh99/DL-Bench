import os
from multiprocessing import cpu_count, Pool
from imagededup.methods import PHash  # You can replace PHash with any other hash method from imagededup
from typing import Dict, Optional, List, Tuple

class EncodeImages:
    def __init__(self):
        self.hasher = PHash()  # Initialize the hash method. Replace PHash with another method if needed.

    def _get_image_files(self, image_dir: str, recursive: bool) -> List[str]:
        """Helper method to get a list of image file paths."""
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    image_files.append(os.path.join(root, file))
            if not recursive:
                break
        return image_files

    def _encode_single_image(self, image_path: str) -> Tuple[str, str]:
        """Helper method to encode a single image and return its filename and hash."""
        try:
            hash_string = self.hasher.encode_image(image_file=image_path)
            return os.path.basename(image_path), hash_string
        except Exception as e:
            print(f"Error encoding {image_path}: {e}")
            return os.path.basename(image_path), None

    def encode_images(self, image_dir: Optional[str] = None, recursive: bool = False, num_enc_workers: int = cpu_count()) -> Dict[str, str]:
        """Generate hashes for all images in a given directory."""
        if image_dir is None:
            raise ValueError("image_dir must be specified.")

        image_files = self._get_image_files(image_dir, recursive)
        if not image_files:
            print("No images found in the specified directory.")
            return {}

        if num_enc_workers == 0:
            # Single-threaded processing
            results = [self._encode_single_image(image_path) for image_path in image_files]
        else:
            # Multi-threaded processing
            with Pool(processes=num_enc_workers) as pool:
                results = pool.map(self._encode_single_image, image_files)

        # Filter out any None results due to errors
        return {filename: hash_string for filename, hash_string in results if hash_string is not None}

