import shutil
import sys
import os
import toml

def copy_and_rename_file(source_file, destination_file):
    # Copy the file to the new location with the new name
    shutil.copy2(source_file, destination_file)
    print(f"File copied to {destination_file}")



def add_bin_section_to_toml(file_path, circuit_name):
    """Adds a second [[bin]] section to the Cargo.toml file."""
    try:
        # Step 1: Read the existing Cargo.toml file using toml (read-write support)
        cargo_data = toml.load(file_path)

        # Step 2: Define the new [[bin]] section to be added
        new_bin_section = {
            "name": circuit_name,  # New binary name
            "path": f"bin/{circuit_name}.rs"  # Path to the binary file
        }

        # Step 3: Ensure the 'bin' section exists (create if it doesn't)
        if 'bin' not in cargo_data:
            cargo_data['bin'] = []
        
        # Step 4: Append the new bin section (this will keep all previous ones intact)
        cargo_data['bin'].append(new_bin_section)

        # Step 5: Write the updated data back to the Cargo.toml file
        with open(file_path, 'w') as f:
            toml.dump(cargo_data, f)

        print(f"Successfully added a second [[bin]] section to {file_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Check if correct number of arguments are passed
    if len(sys.argv) != 2:
        print("Usage: python script.py <circuit_file_name>")
        raise ValueError("Usage: python script.py <circuit_file_name>")

    source_file = "ExpanderCompilerCollection/expander_compiler/tests/testing_template.rs"
    circuit_name = sys.argv[1]
    
    # Source and destination paths from arguments
    destination_file = f"ExpanderCompilerCollection/expander_compiler/bin/{circuit_name}.rs"

    if os.path.exists(destination_file):
        print("Circuit file with this name already exists. Please enter a new file")
        raise ValueError(f"Circuit file with name '{circuit_name}' already exists. Please enter a new file")
    
    # Copy and rename the file
    copy_and_rename_file(source_file, destination_file)

    add_bin_section_to_toml("ExpanderCompilerCollection/expander_compiler/Cargo.toml", circuit_name)

