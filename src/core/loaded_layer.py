from typing import Dict, Any, Optional
import numpy as np
from rasterio.crs import CRS
from affine import Affine


class LoadedLayer:
    """
    Wrapper for raster data as numpy array with spatial metadata.
    
    This class represents a single layer that has been read from disk
    and is ready to be added to a Grid. It maintains the original CRS
    and spatial reference, which Grid will reproject if needed.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        crs: CRS,
        transform: Affine,
        bounds: tuple,
        layer_name: str,
        metadata: Dict[str, Any],
        nodata_value: Optional[float] = None
    ):
        """
        Initialize LoadedLayer.
        
        Args:
            data: Raster data as numpy array (H, W)
            crs: Coordinate reference system
            transform: Affine transform for georeferencing
            bounds: Spatial extent (minx, miny, maxx, maxy)
            layer_name: Name of the layer (e.g., 'cbd', 'elevation')
            metadata: Layer metadata from config
            nodata_value: Value representing missing data
        """
        self.data = data
        self.crs = crs
        self.transform = transform
        self.bounds = bounds
        self.layer_name = layer_name
        self.metadata = metadata
        self.nodata_value = nodata_value
        
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate that data array is 2D."""
        if self.data.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got shape {self.data.shape}"
            )
    
    @classmethod
    def from_base_layer(cls, base_layer):
        """
        Create LoadedLayer from a BaseLayer instance.
        
        Args:
            base_layer: Instance of BaseLayer subclass
            
        Returns:
            LoadedLayer instance
        """
        return base_layer.to_loaded_layer()
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the data array."""
        return self.data.shape
    
    @property
    def layer_type(self) -> str:
        """Get the type of the layer (continuous or categorical)."""
        return self.metadata['type']
    
    def is_categorical(self) -> bool:
        """Check if this layer is categorical."""
        return self.layer_type == 'categorical'
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the layer.
        
        For continuous layers: min, max, mean, median, std, na_count
        For categorical layers: unique_count, na_count
        
        Returns:
            Dictionary of statistics
        """
        mask = ~np.isnan(self.data)
        valid_data = self.data[mask]
        na_count = (~mask).sum()
        
        stats = {
            'layer_name': self.layer_name,
            'layer_type': self.layer_type,
            'shape': self.shape,
            'na_count': int(na_count)
        }
        
        if self.is_categorical():
            unique_values = np.unique(valid_data)
            stats['unique_count'] = len(unique_values)
            stats['unique_values'] = unique_values.tolist()
        else:
            if len(valid_data) > 0:
                stats['min'] = float(np.min(valid_data))
                stats['max'] = float(np.max(valid_data))
                stats['mean'] = float(np.mean(valid_data))
                stats['median'] = float(np.median(valid_data))
                stats['std'] = float(np.std(valid_data))
            else:
                stats['min'] = None
                stats['max'] = None
                stats['mean'] = None
                stats['median'] = None
                stats['std'] = None
        
        return stats
    
    def get_unique_values(self) -> np.ndarray:
        """
        Get unique values in the data (useful for categorical layers).
        
        Returns:
            Array of unique values (excluding NaN)
        """
        mask = ~np.isnan(self.data)
        return np.unique(self.data[mask])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize layer information to dictionary.
        
        Returns:
            Dictionary with layer information (excludes large data array)
        """
        return {
            'layer_name': self.layer_name,
            'layer_type': self.layer_type,
            'shape': self.shape,
            'crs': str(self.crs),
            'bounds': self.bounds,
            'nodata_value': self.nodata_value,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        return (
            f"LoadedLayer("
            f"layer_name='{self.layer_name}', "
            f"shape={self.shape}, "
            f"type='{self.layer_type}'"
            f")"
        )