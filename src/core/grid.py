from typing import Optional, Dict, Any, Tuple, List, Set
from pathlib import Path
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from affine import Affine

from src.config import CHANNEL_ORDER, CHANNEL_METADATA, TARGET_CRS, RESOLUTION
from src.core.loaded_layer import LoadedLayer


class Grid:
    """
    Standardized multi-channel grid for wildfire NCA model.
    
    Manages a (H, W, C) array where all layers are reprojected and resampled
    to a consistent spatial grid. Handles both continuous and categorical data.
    """
    
    def __init__(
        self,
        bounds: tuple,
        resolution: float = RESOLUTION,
        crs: Optional[str] = None,
        channel_names: Optional[List[str]] = None
    ):
        """
        Initialize Grid.
        
        Args:
            bounds: Spatial extent (minx, miny, maxx, maxy) in target CRS
            resolution: Cell size in meters (default from config)
            crs: Target CRS (default from config)
            channel_names: List of channel names (default from config)
        """
        self.bounds = bounds
        self.resolution = resolution
        self.crs = CRS.from_string(crs if crs else TARGET_CRS)
        self.channel_names = channel_names if channel_names else CHANNEL_ORDER.copy()
        
        self.shape = self._calculate_shape()
        self.transform = self._calculate_transform()
        
        num_channels = len(self.channel_names)
        self.data = np.full((self.shape[0], self.shape[1], num_channels), np.nan, dtype=np.float32)
        
        self.channel_metadata = {name: CHANNEL_METADATA[name] for name in self.channel_names}
        self.loaded_channels: Set[str] = set()
        self.categorical_mappings: Dict[str, Dict[int, int]] = {}
    
    def _calculate_shape(self) -> Tuple[int, int]:
        """Calculate grid shape from bounds and resolution."""
        minx, miny, maxx, maxy = self.bounds
        width = int(np.ceil((maxx - minx) / self.resolution))
        height = int(np.ceil((maxy - miny) / self.resolution))
        return (height, width)
    
    def _calculate_transform(self) -> Affine:
        """Calculate affine transform for the grid."""
        minx, miny, maxx, maxy = self.bounds
        return from_bounds(minx, miny, maxx, maxy, self.shape[1], self.shape[0])
    
    def get_channel_idx(self, channel_name: str) -> int:
        """Get the index of a channel in the data array."""
        if channel_name not in self.channel_names:
            raise ValueError(f"Channel '{channel_name}' not in grid")
        return self.channel_names.index(channel_name)
    
    def add_layer(self, loaded_layer: LoadedLayer, channel_name: str) -> None:
        """
        Add a LoadedLayer to the grid.
        
        Reprojects and resamples the layer to match grid's CRS and resolution,
        then inserts into the appropriate channel. For categorical layers,
        applies mapping to convert codes to indices.
        
        Args:
            loaded_layer: LoadedLayer to add
            channel_name: Name of the channel to add to
        """
        if channel_name not in self.channel_names:
            raise ValueError(f"Channel '{channel_name}' not in grid")
        
        if channel_name in self.loaded_channels:
            print(f"Warning: Overwriting existing data in channel '{channel_name}'")
        
        resampled_data = self._resample_to_grid(loaded_layer)
        
        if loaded_layer.is_categorical():
            resampled_data = self._apply_categorical_mapping(resampled_data, channel_name)
        
        channel_idx = self.get_channel_idx(channel_name)
        self.data[:, :, channel_idx] = resampled_data
        self.loaded_channels.add(channel_name)
    
    def _resample_to_grid(self, loaded_layer: LoadedLayer) -> np.ndarray:
        """
        Resample LoadedLayer to match grid's CRS, resolution, and bounds.
        
        Args:
            loaded_layer: LoadedLayer to resample
            
        Returns:
            Resampled data array matching grid shape
        """
        src_crs = loaded_layer.crs
        src_transform = loaded_layer.transform
        src_data = loaded_layer.data
        
        dst_data = np.full(self.shape, np.nan, dtype=np.float32)
        
        resampling_method = Resampling.nearest if loaded_layer.is_categorical() else Resampling.bilinear
        
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=self.transform,
            dst_crs=self.crs,
            resampling=resampling_method,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )
        
        return dst_data
    
    def _apply_categorical_mapping(self, data: np.ndarray, channel_name: str) -> np.ndarray:
        """
        Apply categorical mapping to convert codes to indices.
        
        Args:
            data: Array with original categorical codes
            channel_name: Name of the channel
            
        Returns:
            Array with codes mapped to indices (0 to n-1)
        """
        mask = ~np.isnan(data)
        unique_codes = np.unique(data[mask])
        
        if channel_name not in self.categorical_mappings:
            mapping = {int(code): idx for idx, code in enumerate(sorted(unique_codes))}
            self.categorical_mappings[channel_name] = mapping
        else:
            mapping = self.categorical_mappings[channel_name]
        
        mapped_data = np.full_like(data, np.nan)
        for code, idx in mapping.items():
            mapped_data[data == code] = idx
        
        return mapped_data
    
    def get_channel(self, channel_name: str) -> np.ndarray:
        """
        Get data for a specific channel.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            2D array (H, W) for the channel
        """
        channel_idx = self.get_channel_idx(channel_name)
        return self.data[:, :, channel_idx]
    
    def get_values_at(self, x: float, y: float) -> Dict[str, float]:
        """
        Get values at a geographic coordinate.
        
        Args:
            x: X coordinate (longitude/easting)
            y: Y coordinate (latitude/northing)
            
        Returns:
            Dictionary mapping channel names to values
        """
        row, col = self._coords_to_pixel(x, y)
        
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            raise ValueError(f"Coordinates ({x}, {y}) outside grid bounds")
        
        return {
            name: float(self.data[row, col, idx])
            for idx, name in enumerate(self.channel_names)
        }
    
    def _coords_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert geographic coordinates to pixel row/col."""
        col, row = ~self.transform * (x, y)
        return int(row), int(col)
    
    def summary(self) -> None:
        """Print summary information about the grid."""
        print(f"Grid Summary")
        print(f"  Shape: {self.shape} (H x W)")
        print(f"  Resolution: {self.resolution}m")
        print(f"  CRS: {self.crs}")
        print(f"  Bounds: {self.bounds}")
        print(f"  Total channels: {len(self.channel_names)}")
        print(f"  Loaded channels: {len(self.loaded_channels)}/{len(self.channel_names)}")
        
        if self.loaded_channels:
            print(f"\nLoaded:")
            for name in sorted(self.loaded_channels):
                channel_data = self.get_channel(name)
                na_count = np.isnan(channel_data).sum()
                na_pct = 100 * na_count / channel_data.size
                print(f"    {name}: {na_count} NaN pixels ({na_pct:.1f}%)")
        
        missing = set(self.channel_names) - self.loaded_channels
        if missing:
            print(f"\nMissing:")
            for name in sorted(missing):
                print(f"    {name}")
        
        if self.categorical_mappings:
            print(f"\nCategorical mappings:")
            for name, mapping in self.categorical_mappings.items():
                print(f"    {name}: {len(mapping)} categories")
    
    def save(self, filepath: str) -> None:
        """
        Save grid to disk.
        
        Args:
            filepath: Path to save file (.npz format)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            filepath,
            data=self.data,
            bounds=np.array(self.bounds),
            resolution=self.resolution,
            crs=str(self.crs),
            channel_names=self.channel_names,
            loaded_channels=list(self.loaded_channels),
            categorical_mappings=str(self.categorical_mappings)
        )
        print(f"Grid saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Grid':
        """
        Load grid from disk.
        
        Args:
            filepath: Path to saved grid file
            
        Returns:
            Grid instance
        """
        data = np.load(filepath, allow_pickle=True)
        
        grid = cls(
            bounds=tuple(data['bounds']),
            resolution=float(data['resolution']),
            crs=str(data['crs']),
            channel_names=list(data['channel_names'])
        )
        
        grid.data = data['data']
        grid.loaded_channels = set(data['loaded_channels'])
        grid.categorical_mappings = eval(str(data['categorical_mappings']))
        
        return grid
    
    def __repr__(self) -> str:
        return (
            f"Grid(shape={self.shape}, "
            f"channels={len(self.channel_names)}, "
            f"loaded={len(self.loaded_channels)})"
        )