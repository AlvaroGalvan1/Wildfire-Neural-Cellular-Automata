from pathlib import Path

# Check all __init__.py files
init_files = [
    "src/__init__.py",
    "src/core/__init__.py",
    "src/core/layers/__init__.py",
    "src/utils/__init__.py"
]

for init_file in init_files:
    path = Path(init_file)
    exists = path.exists()
    print(f"{init_file}: {'✓ exists' if exists else '✗ MISSING'}")
    if not exists:
        print(f"  Creating {init_file}...")
        path.touch()