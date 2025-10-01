# src/grid.py
import numpy as np
from rasterio.transform import from_bounds

class Grid:
    """Manages the collection of aligned data layers."""
    def __init__(self, aoi_coords, resolution: int):
        self.layers = {}
        self.target_profile = self._create_target_profile(aoi_coords, resolution)
        print(f"Grid initialized: {self.target_profile['width']}x{self.target_profile['height']} at {resolution}m resolution.")

    def _create_target_profile(self, aoi_coords, resolution):
        minx, miny, maxx, maxy = aoi_coords
        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        # Using a common projected CRS. All layers will be warped to this.
        return {'crs': 'EPSG:3857', 'transform': transform, 'width': width, 'height': height}

    def add_layer(self, layer_object):
        self.layers[layer_object.name] = layer_object
        print(f"Layer '{layer_object.name}' added to the grid.")

    def to_tensor(self):
        if not self.layers:
            return None
        # Stack the layers in the order they were added
        layer_stack = [layer.processed_data for layer in self.layers.values()]
        return np.stack(layer_stack, axis=-1)