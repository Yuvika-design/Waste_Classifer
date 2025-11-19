import os
import shutil
import uuid

# --- CONFIGURATION ---
# IMPORTANT: Update this to the main folder containing your 10 categories (biological, cardboard, etc.)
ORIGINAL_DATA_ROOT = r'archive/images/images' 

# This is the new, flat, Keras-friendly directory that will be created
NEW_FLAT_DATA_ROOT = 'Waste_Classification_FLAT' 
# --- END CONFIGURATION ---

# 1. Create the new root directory if it doesn't exist
os.makedirs(NEW_FLAT_DATA_ROOT, exist_ok=True)
print(f"Created new directory: {NEW_FLAT_DATA_ROOT}")

# 2. Initialize a global file counter for unique naming across ALL categories
# We'll use a unique identifier (UUID) as the base name to be absolutely sure there are no conflicts.
file_counter = 0

# Define supported image extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

print("--- Starting File Restructuring and Renaming ---")

# os.walk() is perfect for traversing the entire directory tree recursively
for root, dirs, files in os.walk(ORIGINAL_DATA_ROOT):
    
    # Check if the current directory is one of the inner folders we want to merge (default or real_world)
    # The parent directory of 'default' or 'real_world' is the actual class name (e.g., 'biological')
    if os.path.basename(root) in ['default', 'real_world']:
        
        # Get the class name (the folder above 'default'/'real_world')
        # os.path.dirname(root) gets the parent folder. os.path.basename() extracts its name.
        class_name = os.path.basename(os.path.dirname(root))
        
        # 3. Create the destination class folder in the new flat structure
        destination_class_path = os.path.join(NEW_FLAT_DATA_ROOT, class_name)
        os.makedirs(destination_class_path, exist_ok=True)
        
        # 4. Process and move each file
        for filename in files:
            # Check if it's a valid image file
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                
                # Full path to the original file
                src_path = os.path.join(root, filename)
                
                # Get the file extension (e.g., .jpg)
                _, file_extension = os.path.splitext(filename)
                
                # Generate a unique name using a counter and UUID
                unique_name = f"{class_name}_{file_counter}_{uuid.uuid4().hex[:8]}{file_extension}"
                
                # Full path to the new file
                dst_path = os.path.join(destination_class_path, unique_name)
                
                # Copy the file to the new location with the unique name
                shutil.copy(src_path, dst_path)
                
                file_counter += 1

print("\n--- Process Complete ---")
print(f"Total files moved and renamed: {file_counter}")
print(f"The Keras-ready directory is: {NEW_FLAT_DATA_ROOT}")