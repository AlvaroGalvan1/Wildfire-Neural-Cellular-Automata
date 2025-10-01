# src/layer.py
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

class LoadedLayer:
    """A container for a single data layer, responsible for its own processing."""
    def __init__(self, name: str, layer_type: str, source_path: str):
        self.name = name
        self.layer_type = layer_type
        self.source_path = source_path
        self.processed_data = None
        self.metadata = {}

    def process(self, target_profile: dict):
        """
        Loads, resamples, and processes the layer to match the target grid profile.
        """
        print(f"Processing layer: {self.name}...")
        
        with rasterio.open(self.source_path) as src:
            # Create an empty array with the target dimensions
            destination = np.zeros((target_profile['height'], target_profile['width']), dtype=src.profile['dtype'])
            
            resampling_method = None
            if self.layer_type == 'continuous':
                resampling_method = Resampling.bilinear
            elif self.layer_type == 'categorical':
                resampling_method = Resampling.nearest
            else:
                raise ValueError(f"Unknown layer type: {self.layer_type}")

            # Reproject/resample the source data to the target profile
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_profile['transform'],
                dst_crs=target_profile['crs'],
                resampling=resampling_method
            )
            
            # --- Post-Resampling Processing ---
            if self.layer_type == 'continuous':
                print("  - Layer type: Continuous. Normalizing to [0, 1].")
                # Normalize canopy cover by dividing by 100
                destination = destination / 100.0
                destination[destination < 0] = 0 # Ensure no negative values
                self.processed_data = destination

            elif self.layer_type == 'categorical':
                print("  - Layer type: Categorical. Performing integer encoding.")
                unique_vals = np.unique(destination)
                print(f"  - Found {len(unique_vals)} unique categories.")
                
                # Create the mapping from original code to 0-indexed integer
                encoding_map = {val: i for i, val in enumerate(unique_vals)}
                self.metadata['encoding_map'] = encoding_map
                
                # Apply the mapping
                encoded_data = np.zeros_like(destination)
                for original_val, encoded_val in encoding_map.items():
                    encoded_data[destination == original_val] = encoded_val
                
                self.processed_data = encoded_data.astype(np.int32)
                print(f"  - Encoded values to [0, {len(unique_vals) - 1}].")
        
        print(f"  - Resampled to {self.processed_data.shape}.")