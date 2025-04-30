import cv2
import numpy as np
import glob
import os

def create_one_row_strip(
    image_folder="images/", 
    output_file="one_row_strip.jpg", 
    thumb_width=320, 
    thumb_height=180
):
    # Get list of image files (sorted to keep sequence order)
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

    # Option 1: Resize each image to a specific thumbnail size
    resized_images = []
    for idx, img in enumerate(images):
        resized = cv2.resize(img, (thumb_width, thumb_height))
        resized_images.append(resized)
        print(f"Image {idx} resized to: {resized.shape}")
    
    # Option 2: Alternatively, use a common height (uncomment below to use that option)
    common_height = min(img.shape[0] for img in images)
    resized_images = []
    for idx, img in enumerate(images):
        scale = common_height / img.shape[0]
        new_w = int(img.shape[1] * scale)
        resized = cv2.resize(img, (new_w, common_height))
        resized_images.append(resized)
        print(f"Image {idx} resized to: {resized.shape}")
    
    # Concatenate images horizontally (one row)
    one_row = np.hstack(resized_images)
    
    # Save the final one-row composite image.
    success = cv2.imwrite(output_file, one_row)
    if success:
        print(f"One-row strip saved as {output_file}")
    else:
        print("Failed to save one-row strip.")
    
    return one_row

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    
    # Adjust thumb_width and thumb_height as needed.
    result = create_one_row_strip(thumb_width=320, thumb_height=180)
    if result is not None:
        cv2.namedWindow("One Row Strip", cv2.WINDOW_NORMAL)
        cv2.imshow("One Row Strip", result)
        print("Displaying one row strip. Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
