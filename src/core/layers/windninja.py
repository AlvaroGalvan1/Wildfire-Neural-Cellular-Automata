from typing import Tuple, Dict, Any
from pathlib import Path
import numpy as np
import rasterio

from src.core.layers.base_layer import BaseLayer


class WindNinjaLayer(BaseLayer):
    """
    Layer class for WindNinja wind simulation data.
    
    Handles reading WindNinja .tif files for specific timestamps.
    Each timestamp has wind_speed and wind_direction outputs.
    """
    
    def __init__(self, data_dir: str, variable_name: str, timestamp: str):
        """
        Initialize WindNinjaLayer.
        
        Args:
            data_dir: Path to WindNinja data directory
            variable_name: Name of the variable ('wind_speed' or 'wind_direction')
            timestamp: Timestamp string (e.g., '2024-07-15_12-00')
        """
        from src.config import WINDNINJA_VARIABLES
        
        if variable_name not in WINDNINJA_VARIABLES:
            raise ValueError(
                f"Unknown WindNinja variable: {variable_name}. "
                f"Must be one of: {list(WINDNINJA_VARIABLES.keys())}"
            )
        
        filename_template = WINDNINJA_VARIABLES[variable_name]
        filename = filename_template.format(timestamp=timestamp)
        file_path = Path(data_dir) / filename
        
        self.timestamp = timestamp
        
        super().__init__(str(file_path), variable_name)
    
    def read(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read WindNinja .tif file and return data with spatial metadata.
        
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
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"layer_name='{self.layer_name}', "
            f"timestamp='{self.timestamp}', "
            f"file_path='{self.file_path}'"
            f")"
        )