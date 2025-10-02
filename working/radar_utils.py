# src/radar_utils.py
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import os

def preprocess_frame(db_frame, min_val, max_val, output_path=None):
    """
    Normalizes a dB-scaled frame and converts it to a colormapped PIL Image.
    """

    # normalized_frame = (db_frame - min_val) / (max_val - min_val)
    # normalized_frame = np.clip(normalized_frame, 0, 1)
    rd_map_clipped = np.clip(db_frame, min_val, max_val)
    denominator = max_val - min_val
    if denominator < 1e-6: # Use a small epsilon for floating point safety
        denominator = 1e-6 
    rd_map_normalized = (rd_map_clipped - min_val) / denominator


    # Apply colormap and convert to PIL Image
    colormap = cm.get_cmap('viridis')
    rgb_frame = colormap(rd_map_normalized)[:, :, :3]
    image = Image.fromarray((rgb_frame * 255).astype(np.uint8))
    # output_path="/home/huy/seb_paper/clip_sim2real/zero_shot_sim_to_real/train_seen"
    # --- NEW: Save the image if an output path is provided ---
    if output_path:
        # print(f"Saving debug image to {output_path}...")
        try:
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Save the PIL Image
            image.save(output_path)
        except Exception as e:
            # Add a warning if saving fails for any reason
            print(f"Warning: Could not save debug image to {output_path}. Error: {e}")

    return image