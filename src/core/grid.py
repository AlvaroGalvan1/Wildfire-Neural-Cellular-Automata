from typing import Optional, Dict, Any, Tuple, List, Set
from pathlib import Path
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
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
        channel_names: Optional[List[str]] = None,
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
        self.timestamp: Optional[str] = None  # <-- Added here

    # ---------------------------- Shape & Transform ---------------------------- #
    
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

    # ---------------------------- Channels ---------------------------- #
    
    def get_channel_idx(self, channel_name: str) -> int:
        """Get the index of a channel in the data array."""
        if channel_name not in self.channel_names:
            raise ValueError(f"Channel '{channel_name}' not in grid")
        return self.channel_names.index(channel_name)
    
    def add_layer(self, loaded_layer: LoadedLayer, channel_name: str) -> None:
        """
        Add a LoadedLayer to the grid.
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
        """Resample LoadedLayer to match grid CRS, resolution, and bounds."""
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
        """Apply categorical mapping to convert codes to indices."""
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

    # ---------------------------- Access Methods ---------------------------- #
    
    def get_channel(self, channel_name: str) -> np.ndarray:
        """Get data for a specific channel."""
        channel_idx = self.get_channel_idx(channel_name)
        return self.data[:, :, channel_idx]
    
    def get_values_at(self, x: float, y: float) -> Dict[str, float]:
        """Get values at a geographic coordinate."""
        row, col = self._coords_to_pixel(x, y)
        
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            raise ValueError(f"Coordinates ({x}, {y}) outside grid bounds")
        
        return {
            name: float(self.data[row, col, idx])
            for idx, name in enumerate(self.channel_names)
        }

    def set_timestamp(self, timestamp: str) -> None:
        """
        Set timestamp for dynamic data (wind variables).
        
        Args:
            timestamp: Timestamp string (e.g., '2024-07-15_12-00')
        """
        self.timestamp = timestamp
    
    def _coords_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert geographic coordinates to pixel row/col."""
        col, row = ~self.transform * (x, y)
        return int(row), int(col)

    # ---------------------------- Summary & IO ---------------------------- #
    
    def summary(self) -> None:
        """Print summary information about the grid."""
        print(f"Grid Summary")
        print(f"  Shape: {self.shape} (H x W)")
        print(f"  Resolution: {self.resolution}m")
        print(f"  CRS: {self.crs}")
        print(f"  Bounds: {self.bounds}")
        print(f"  Total channels: {len(self.channel_names)}")
        print(f"  Loaded channels: {len(self.loaded_channels)}/{len(self.channel_names)}")
        
        if self.timestamp:
            print(f"  Timestamp: {self.timestamp}")
        
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
        """Save grid to disk (.npz format)."""
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
            categorical_mappings=str(self.categorical_mappings),
            timestamp=self.timestamp if self.timestamp else ""
        )
        print(f"Grid saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Grid':
        """Load grid from disk."""
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
        grid.timestamp = str(data['timestamp']) if data['timestamp'] else None
        
        return grid

    def __repr__(self) -> str:
        timestamp_str = f", timestamp='{self.timestamp}'" if self.timestamp else ""
        return (
            f"Grid(shape={self.shape}, "
            f"channels={len(self.channel_names)}, "
            f"loaded={len(self.loaded_channels)}{timestamp_str})"
        )
