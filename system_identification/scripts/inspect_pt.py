import torch
import sys
import os


def print_structure(data, indent=0, max_items=100):
    prefix = " " * indent
    if isinstance(data, dict):
        print(f"{prefix}dict with {len(data)} keys:")
        for i, (k, v) in enumerate(data.items()):
            if i >= max_items:
                print(f"{prefix}  ... ({len(data) - max_items} more keys)")
                break
            print(f"{prefix}  Key '{k}':")
            print_structure(v, indent + 4)
    elif isinstance(data, list):
        print(f"{prefix}list with {len(data)} items:")
        for i, item in enumerate(data):
            if i >= max_items:
                print(f"{prefix}  ... ({len(data) - max_items} more items)")
                break
            print_structure(item, indent + 2)
    elif isinstance(data, tuple):
        print(f"{prefix}tuple with {len(data)} items:")
        for i, item in enumerate(data):
            if i >= max_items:
                print(f"{prefix}  ... ({len(data) - max_items} more items)")
                break
            print_structure(item, indent + 2)
    elif torch.is_tensor(data):
        print(
            f"{prefix}Tensor shape={data.shape}, dtype={data.dtype}, device={data.device}"
        )
        # Print a few values
        flat_data = data.flatten()
        values_to_print = flat_data[: min(5, flat_data.numel())].tolist()
        print(f"{prefix}  First few values: {values_to_print}")
    elif hasattr(data, "shape"):  # numpy array
        print(f"{prefix}Array shape={data.shape}, dtype={data.dtype}")
        flat_data = data.flatten()
        values_to_print = flat_data[: min(5, flat_data.size)].tolist()
        print(f"{prefix}  First few values: {values_to_print}")
    else:
        print(f"{prefix}{type(data)}: {data}")


def main():
    file_path = "chirp_data.pt"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    try:
        data = torch.load(file_path, map_location=torch.device("cpu"))
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Structure:")
    print_structure(data)


if __name__ == "__main__":
    main()
