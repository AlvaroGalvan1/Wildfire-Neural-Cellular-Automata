from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
from rasterio.crs import CRS
from affine import Affine

from src.config import CHANNEL_METADATA


class BaseLayer(ABC):
    """
    Abstract base class for all data source layers.
    
    Defines the interface that all layer types (Landfire, USGS, WindNinja)
    must implement. Handles common validation and metadata loading.
    """
    
    def __init__(self, file_path: str, layer_name: str):
        """
        Initialize base layer.
        
        Args:
            file_path: Path to the raster file
            layer_name: Name of the layer (must exist in CHANNEL_METADATA)
        """
        self.file_path = Path(file_path)
        self.layer_name = layer_name
        
        self._validate_file_path()
        self._validate_layer_name()
        
        self.metadata = CHANNEL_METADATA[layer_name]
    
    def _validate_file_path(self) -> None:
        """Validate that the file exists."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")
    
    def _validate_layer_name(self) -> None:
        """Validate that the layer name exists in configuration."""
        if self.layer_name not in CHANNEL_METADATA:
            raise ValueError(
                f"Unknown layer name: {self.layer_name}. "
                f"Must be one of: {list(CHANNEL_METADATA.keys())}"
            )
    
    @abstractmethod
    def read(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read the raster file and return data with spatial metadata.
        
        Must be implemented by subclasses.
        
        Returns:
            Tuple containing:
                - data: numpy array of raster values (H, W)
                - raster_info: dict with keys:
                    - 'crs': CRS object
                    - 'transform': Affine transform
                    - 'bounds': tuple (minx, miny, maxx, maxy)
                    - 'nodata': nodata value or None
        """
        pass
    
    def to_loaded_layer(self):
        """
        Read the data and wrap it in a LoadedLayer object.
        
        This method is the same for all layer types, so it's implemented
        in the base class rather than being abstract.
        
        Returns:
            LoadedLayer object containing the raster data and metadata
        """
        from src.core.loaded_layer import LoadedLayer
        
        data, raster_info = self.read()
        
        return LoadedLayer(
            data=data,
            crs=raster_info['crs'],
            transform=raster_info['transform'],
            bounds=raster_info['bounds'],
            layer_name=self.layer_name,
            metadata=self.metadata,
            nodata_value=raster_info['nodata']
        )
    
    def get_layer_type(self) -> str:
        """
        Get the type of the layer (continuous or categorical).
        
        Returns:
            'continuous' or 'categorical'
        """
        return self.metadata['type']
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"layer_name='{self.layer_name}', "
            f"file_path='{self.file_path}'"
            f")"
        )