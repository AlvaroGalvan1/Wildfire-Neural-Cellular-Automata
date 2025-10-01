# src/data_ingester.py
import os

def find_raw_data_path(product_code: str) -> str:
    """
    Finds the path to a raw data file based on the product code.
    This is a simple placeholder for a real data downloading pipeline.
    
    Args:
        product_code (str): The LandFire product code (e.g., '240CC').

    Returns:
        str: The full path to the .tif file.
    """
    raw_data_dir = 'data/raw'
    for filename in os.listdir(raw_data_dir):
        # Find the first .tif file that contains the product code
        if product_code in filename and filename.endswith('.tif'):
            print(f"Found pre-existing file: {filename}")
            return os.path.join(raw_data_dir, filename)
    
    raise FileNotFoundError(f"No .tif file found for product '{product_code}' in '{raw_data_dir}'. Please download it first.")