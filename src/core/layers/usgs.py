from typing import Tuple, Dict, Any
import numpy as np
import rasterio

from src.core.layers.base_layer import BaseLayer


class USGSLayer(BaseLayer):
    """
    Layer class for USGS 3DEP topographic data.
    
    Handles reading USGS .tif files including elevation, slope, and aspect.
    These products are typically at 10m resolution in WGS84 (lat/lon).
    """
    
    def __init__(self, file_path: str, variable_name: str):
        """
        Initialize USGSLayer.
        
        Args:
            file_path: Path to the USGS .tif file
            variable_name: Name of the variable (e.g., 'elevation', 'slope', 'aspect')
        """
        super().__init__(file_path, variable_name)
    
    def read(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read USGS .tif file and return data with spatial metadata.
        
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