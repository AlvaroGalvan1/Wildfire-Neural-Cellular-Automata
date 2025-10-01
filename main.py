# main.py
from src.config import STUDY_AREA_COORDS, TARGET_RESOLUTION
from src.grid import Grid
from src.layer import LoadedLayer
from src.data_ingester import find_raw_data_path
import numpy as np

def main():
    print("\n--- Wildfire NCA MVP ---")

    # 1. Initialize the Grid with our desired profile
    print("\n[STEP 1] Initializing Grid...")
    grid = Grid(aoi_coords=STUDY_AREA_COORDS, resolution=TARGET_RESOLUTION)

    # 2. Define the layers we want to ingest
    layers_to_load = [
        {'name': 'CanopyCover', 'type': 'continuous', 'code': '240CC'},
        {'name': 'BPS',         'type': 'categorical','code': '200BPS'}
    ]

    # 3. Process each layer and add it to the grid
    for i, layer_info in enumerate(layers_to_load):
        print(f"\n[STEP {2+i}] Ingesting raw data for {layer_info['name']} ({layer_info['code']})...")
        
        # Find the raw data file
        raw_path = find_raw_data_path(layer_info['code'])
        
        # Create a layer object
        layer = LoadedLayer(name=layer_info['name'], layer_type=layer_info['type'], source_path=raw_path)
        
        # Process the layer to match the grid's profile
        layer.process(target_profile=grid.target_profile)
        
        # Add the processed layer to the grid
        grid.add_layer(layer)
        
    # 4. Assemble the final multi-channel tensor
    print(f"\n[STEP {2+len(layers_to_load)}] Assembling final tensor...")
    final_tensor = grid.to_tensor()
    
    # 5. Verify and save the output
    if final_tensor is not None:
        print("\n--- SUCCESS! ---")
        print(f"Final tensor created with shape HxWxC: {final_tensor.shape}")
        
        np.save('final_tensor.npy', final_tensor)
        print("Final tensor saved to 'final_tensor.npy'")
        
        # Verify layer data types and ranges
        cc_data = grid.layers['CanopyCover'].processed_data
        bps_data = grid.layers['BPS'].processed_data
        print(f"\nCanopy Cover (Channel 0):\n  - DType: {cc_data.dtype}, Min: {cc_data.min():.4f}, Max: {cc_data.max():.4f}")
        print(f"\nBPS (Channel 1):\n  - DType: {bps_data.dtype}, Min: {bps_data.min()}, Max: {bps_data.max()}")
        print(f"  - Unique encoded categories: {len(np.unique(bps_data))}")

if __name__ == "__main__":
    main()
