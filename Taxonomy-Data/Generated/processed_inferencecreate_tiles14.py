import numpy as np
import cv2

def create_tiles(images, grid_size=None, tile_size=None, scaling_method='min', 
                 padding_color=(0, 0, 0), tile_margin=0, margin_color=(255, 255, 255)):
    if not images:
        raise ValueError("The list of images is empty.")
    
    num_images = len(images)
    
    # Determine grid size if not provided
    if grid_size is None:
        grid_rows = grid_cols = int(np.ceil(np.sqrt(num_images)))
    else:
        grid_rows, grid_cols = grid_size
    
    if num_images > grid_rows * grid_cols:
        raise ValueError("The number of images exceeds the grid size.")
    
    # Determine tile size if not provided
    if tile_size is None:
        # Use the size of the first image as the default tile size
        tile_height, tile_width = images[0].shape[:2]
    else:
        tile_height, tile_width = tile_size
    
    # Resize images according to the scaling method
    resized_images = []
    for img in images:
        if scaling_method == 'min':
            scale_factor = min(tile_width / img.shape[1], tile_height / img.shape[0])
        elif scaling_method == 'max':
            scale_factor = max(tile_width / img.shape[1], tile_height / img.shape[0])
        elif scaling_method == 'avg':
            scale_factor = (tile_width / img.shape[1] + tile_height / img.shape[0]) / 2
        else:
            raise ValueError("Invalid scaling method. Choose 'min', 'max', or 'avg'.")
        
        new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        
        # Create a new image with the desired tile size and padding color
        tile_img = np.full((tile_height, tile_width, 3), padding_color, dtype=np.uint8)
        
        # Center the resized image in the tile
        y_offset = (tile_height - resized_img.shape[0]) // 2
        x_offset = (tile_width - resized_img.shape[1]) // 2
        tile_img[y_offset:y_offset+resized_img.shape[0], x_offset:x_offset+resized_img.shape[1]] = resized_img
        
        resized_images.append(tile_img)
    
    # Create the final grid image with margins
    grid_height = grid_rows * tile_height + (grid_rows - 1) * tile_margin
    grid_width = grid_cols * tile_width + (grid_cols - 1) * tile_margin
    grid_image = np.full((grid_height, grid_width, 3), margin_color, dtype=np.uint8)
    
    # Place each tile in the grid
    for idx, tile_img in enumerate(resized_images):
        row = idx // grid_cols
        col = idx % grid_cols
        y_start = row * (tile_height + tile_margin)
        x_start = col * (tile_width + tile_margin)
        grid_image[y_start:y_start+tile_height, x_start:x_start+tile_width] = tile_img
    
    return grid_image