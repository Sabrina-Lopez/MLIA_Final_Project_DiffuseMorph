import os
import argparse

def rename_files_by_class(dataroot):
    """
    Renames files in a dataset directory by prepending the class name (folder name).

    For a structure like:
    dataroot/
        circle/
            0.png
            1.png
        square/
            0.png

    It will rename the files to:
    dataroot/
        circle/
            circle_0.png
            circle_1.png
        square/
            square_0.png
    """
    print(f"Starting file renaming process for directory: {dataroot}")

    if not os.path.isdir(dataroot):
        print(f"Error: Directory not found at {dataroot}")
        return

    try:
        class_folders = [d for d in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, d))]
    except FileNotFoundError:
        print(f"Error: Could not access directory {dataroot}")
        return

    total_renamed = 0
    for class_name in class_folders:
        # For MNIST, use the number before '_affined'
        simple_class_name = class_name.split('_')[0]
        class_path = os.path.join(dataroot, class_name)
        print(f"\nProcessing class: {class_name} (using prefix '{simple_class_name}')")

        try:
            # Walk through subdirectories like 'type 1', 'type 2' etc.
            for root, _, files in os.walk(class_path):
                renamed_in_folder = 0
                for filename in files:
                    # Check if the file is already renamed to avoid double-renaming
                    if filename.startswith(simple_class_name + '_'):
                        continue

                    old_filepath = os.path.join(root, filename)
                    new_filename = f"{simple_class_name}_{filename}"
                    new_filepath = os.path.join(root, new_filename)

                    try:
                        os.rename(old_filepath, new_filepath)
                        print(f"  - Renamed: {filename} -> {new_filename} in {os.path.basename(root)}")
                        renamed_in_folder += 1
                    except OSError as e:
                        print(f"  - Error renaming {filename}: {e}")
                
                if renamed_in_folder > 0:
                    total_renamed += renamed_in_folder
        except FileNotFoundError:
            print(f"  Warning: Could not access class directory {class_path}")
            continue

    if total_renamed > 0:
        print(f"\nFinished. Renamed a total of {total_renamed} files.")
    else:
        print("\nFinished. No files needed renaming.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename dataset files by prepending their class name.")
    parser.add_argument(
        '--dataroot', 
        type=str, 
        required=True, 
        help="The root directory of the dataset (e.g., './datasets/google_quickdraw')"
    )
    args = parser.parse_args()
    
    rename_files_by_class(args.dataroot)
