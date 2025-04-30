import cv2
import numpy as np
import glob
import os
import math

def create_mosaic(
    image_folder="images/", 
    output_file="mosaic.jpg", 
    thumb_width=320, 
    thumb_height=180, 
    grid_cols=8
):
    # Get list of JPEG files (sorted to keep sequence order)
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if not image_files:
        print("No images found in the folder.")
        return None

    print("Found image files:", image_files)
    images = []
    # Load each image
    for file in image_files:
        img = cv2.imread(file)
        if img is None:
            print(f"Failed to load {file}. Skipping.")
            continue
        images.append(img)
        print(f"Loaded {file} with shape: {img.shape}")

    if not images:
        print("No images loaded successfully.")
        return None

    num_images = len(images)
    grid_rows = math.ceil(num_images / grid_cols)
    print(f"Arranging {num_images} images into {grid_rows} rows and {grid_cols} columns.")

    # Create thumbnails and arrange them in a list
    thumbnails = []
    for idx, img in enumerate(images):
        thumb = cv2.resize(img, (thumb_width, thumb_height))
        thumbnails.append(thumb)
        print(f"Image {idx} resized to thumbnail: {thumb.shape}")

    # Create a blank canvas for the mosaic
    canvas_width = grid_cols * thumb_width
    canvas_height = grid_rows * thumb_height
    mosaic = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place thumbnails into the mosaic grid
    for i, thumb in enumerate(thumbnails):
        row = i // grid_cols
        col = i % grid_cols
        x = col * thumb_width
        y = row * thumb_height
        mosaic[y:y+thumb_height, x:x+thumb_width] = thumb
        print(f"Placed image {i} at position: row {row}, col {col} (x={x}, y={y})")

    # Save the final mosaic image
    success = cv2.imwrite(output_file, mosaic)
    if success:
        print(f"Mosaic saved as {output_file}")
    else:
        print("Failed to save mosaic.")

    return mosaic

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    
    # Adjust parameters as needed:
    # thumb_width and thumb_height control the size of each thumbnail.
    # grid_cols sets how many images per row.
    result = create_mosaic(thumb_width=320, thumb_height=180, grid_cols=8)
    if result is not None:
        cv2.namedWindow("Mosaic", cv2.WINDOW_NORMAL)
        cv2.imshow("Mosaic", result)
        print("Displaying mosaic. Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
