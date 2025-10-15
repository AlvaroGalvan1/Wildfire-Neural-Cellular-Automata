"""
Central configuration for wildfire NCA data pipeline.
Defines spatial parameters, channel metadata, and data source mappings.
"""

SAMPLE_001_BOUNDS_LATLON = (-119.363, 37.13, -119.137, 37.31)

TARGET_CRS = "ESRI:102008"

RESOLUTION = 30

CHANNEL_ORDER = [
    "cbd",
    "cbh",
    "cc",
    "ch",
    "fbfm40",
    "dist",
    "elevation",
    "slope",
    "aspect",
    "wind_speed",
    "wind_direction"
]

CHANNEL_METADATA = {
    "cbd": {
        "type": "continuous",
        "units": "kg/m³",
        "description": "Forest Canopy Bulk Density",
        "expected_range": (0, 30),
        "source": "landfire"
    },
    "cbh": {
        "type": "continuous",
        "units": "m",
        "description": "Forest Canopy Base Height",
        "expected_range": (0, 100),
        "source": "landfire"
    },
    "cc": {
        "type": "continuous",
        "units": "%",
        "description": "Forest Canopy Cover",
        "expected_range": (0, 100),
        "source": "landfire"
    },
    "ch": {
        "type": "continuous",
        "units": "m",
        "description": "Forest Canopy Height",
        "expected_range": (0, 400),
        "source": "landfire"
    },
    "fbfm40": {
        "type": "categorical",
        "units": "code",
        "description": "Fire Behavior Fuel Model 40 (Scott & Burgan)",
        "num_classes": None,
        "source": "landfire"
    },
    "dist": {
        "type": "categorical",
        "units": "code",
        "description": "Fire History Disturbance 2024",
        "num_classes": None,
        "source": "landfire"
    },
    "elevation": {
        "type": "continuous",
        "units": "m",
        "description": "Elevation above sea level",
        "expected_range": (0, 5000),
        "source": "usgs"
    },
    "slope": {
        "type": "continuous",
        "units": "degrees",
        "description": "Terrain slope",
        "expected_range": (0, 90),
        "source": "usgs"
    },
    "aspect": {
        "type": "continuous",
        "units": "degrees",
        "description": "Terrain aspect (direction of slope)",
        "expected_range": (0, 360),
        "source": "usgs"
    },
    "wind_speed": {
        "type": "continuous",
        "units": "m/s",
        "description": "Wind speed",
        "expected_range": (0, 50),
        "source": "windninja"
    },
    "wind_direction": {
        "type": "continuous",
        "units": "degrees",
        "description": "Wind direction (meteorological convention: direction FROM)",
        "expected_range": (0, 360),
        "source": "windninja"
    }
}

LANDFIRE_VARIABLES = {
    "cbd": "240CBD.tif",
    "cbh": "240CBH.tif",
    "cc": "240CC.tif",
    "ch": "240CH.tif",
    "fbfm40": "240FBFM40.tif",
    "dist": "DIST2024.tif"
}

USGS_VARIABLES = {
    "elevation": "elevation_10m_sample001.tif",
    "slope": "slope.tif",
    "aspect": "aspect.tif"
}

SAMPLE_001_WINDNINJA_TIMESTAMPS = [
    "2024-07-15_06-00",
    "2024-07-15_09-00",
    "2024-07-15_12-00",
    "2024-07-15_15-00"
]

DEFAULT_TIMESTAMP = "2024-07-15_12-00"

WINDNINJA_VARIABLES = {
    "wind_speed": "wind_speed_{timestamp}.tif",
    "wind_direction": "wind_direction_{timestamp}.tif"
}

FBFM40_MAPPING = {}

FBFM40_REVERSE_MAPPING = {}

DIST2024_MAPPING = {}

DIST2024_REVERSE_MAPPING = {}

def get_channel_index(channel_name: str) -> int:
    """Get the index of a channel in the standardized grid."""
    return CHANNEL_ORDER.index(channel_name)

def is_categorical(channel_name: str) -> bool:
    """Check if a channel is categorical."""
    return CHANNEL_METADATA[channel_name]["type"] == "categorical"

def get_num_channels() -> int:
    """Get total number of channels."""
    return len(CHANNEL_ORDER)