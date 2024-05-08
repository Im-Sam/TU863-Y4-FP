import os

# Function to rename files in a directory
def rename_files(directory_path, new_name_base):
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    for i, file in enumerate(files, start=1):
        extension = os.path.splitext(file)[1]
        new_name = f"{new_name_base}{i}{extension}"
        old_file = os.path.join(directory_path, file)
        new_file = os.path.join(directory_path, new_name)
        os.rename(old_file, new_file)
        print(f"Renamed {file} to {new_name}")

# Paths to the directories
ai_generated_path = './dataset/ai_generated'  # Update this path
real_path = './dataset/real'  # Update this path

# Rename files in the ai_generated directory
rename_files(ai_generated_path, 'ai_image')

# Rename files in the real directory
rename_files(real_path, 'real_image')

print("All files have been renamed.")
