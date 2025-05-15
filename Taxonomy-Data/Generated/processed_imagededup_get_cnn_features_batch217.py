import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import PurePath
from typing import Optional, Dict
import numpy as np
import logging

def _get_cnn_features_batch(image_dir: PurePath, recursive: Optional[bool] = False, num_workers: int = 0) -> Dict[str, np.ndarray]:
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    dataset = ImageFolder(root=str(image_dir), transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    # Load a pre-trained CNN model
    model = models.resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Remove the final classification layer to get feature vectors
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # Dictionary to store the results
    features_dict = {}

    # Process each batch
    for batch_idx, (images, _) in enumerate(data_loader):
        try:
            with torch.no_grad():
                # Forward pass to get features
                features = model(images).squeeze().numpy()

            # Map filenames to features
            for i, (path, _) in enumerate(data_loader.dataset.samples[batch_idx * data_loader.batch_size: (batch_idx + 1) * data_loader.batch_size]):
                filename = PurePath(path).name
                features_dict[filename] = features[i]

            logger.info(f"Processed batch {batch_idx + 1}/{len(data_loader)}")

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")

    return features_dict