import os
import shutil
import glob
import numpy as np

def move_subset_to_input(source_dir, target_base_dir, subset_percentage=0.04):
    # Ensure the base input directory exists
    os.makedirs(target_base_dir, exist_ok=True)
    
    # Subdirectories for the input, matching the structure of the source
    ai_gen_input_dir = os.path.join(target_base_dir, 'ai_generated')
    real_input_dir = os.path.join(target_base_dir, 'real')
    
    os.makedirs(ai_gen_input_dir, exist_ok=True)
    os.makedirs(real_input_dir, exist_ok=True)
    
    # Handle 'ai_generated' directory
    ai_gen_dir = os.path.join(source_dir, 'ai_generated')
    real_dir = os.path.join(source_dir, 'real')
    
    # Function to randomly move files
    def move_files(source, destination, percentage):
        all_files = glob.glob(os.path.join(source, '*'))
        np.random.shuffle(all_files)
        num_to_move = int(len(all_files) * percentage)
        files_to_move = all_files[:num_to_move]
        for file in files_to_move:
            shutil.move(file, destination)

    # Move a subset of files from 'ai_generated' and 'real'
    move_files(ai_gen_dir, ai_gen_input_dir, subset_percentage)
    move_files(real_dir, real_input_dir, subset_percentage)

    print(f"Moved {subset_percentage * 100}% of files from '{ai_gen_dir}' to '{ai_gen_input_dir}'")
    print(f"Moved {subset_percentage * 100}% of files from '{real_dir}' to '{real_input_dir}'")

# Example usage
source_directory = './dataset/'  # Adjust this path to your dataset's location
destination_directory = './input/'  # Destination base directory for 'input'

move_subset_to_input(source_directory, destination_directory)
