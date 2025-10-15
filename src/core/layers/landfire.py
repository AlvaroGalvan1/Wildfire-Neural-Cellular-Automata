from typing import Tuple, Dict, Any
import numpy as np
import rasterio
from rasterio.crs import CRS

from src.core.layers.base_layer import BaseLayer


class LandfireLayer(BaseLayer):
    """
    Layer class for Landfire data products.
    
    Handles reading Landfire .tif files including both continuous variables
    (CBD, CBH, CC, CH) and categorical variables (FBFM40, DIST2024).
    """
    
    def __init__(self, file_path: str, variable_name: str):
        """
        Initialize LandfireLayer.
        
        Args:
            file_path: Path to the Landfire .tif file
            variable_name: Name of the variable (e.g., 'cbd', 'fbfm40')
        """
        super().__init__(file_path, variable_name)
    
    def read(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read Landfire .tif file and return data with spatial metadata.
        
        Returns:
            Tuple containing:
                - data: numpy array with nodata replaced by np.nan
                - raster_info: dict with crs, transform, bounds, nodata
        """
        with rasterio.open(self.file_path) as src:
            data = src.read(1)
            
            nodata = src.nodata
            if nodata is not None:
                data = data.astype(float)
                data[data == nodata] = np.nan
            else:
                data = data.astype(float)
            
            raster_info = {
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'nodata': nodata
            }
        
        return data, raster_info