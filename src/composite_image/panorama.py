import cv2
import numpy as np
import glob
import os

def create_movie_strip(image_folder="images/", output_file="movie_strip.jpg", final_height=800):
    # Get list of image files (sorted to keep the sequence order)
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
        print(f"Loaded {file} with shape: {img.shape}")
        images.append(img)

    if not images:
        print("No images loaded successfully.")
        return None

    # Use the specified final_height for all images
    print("Resizing images to a height of:", final_height)
    resized_images = []
    for idx, img in enumerate(images):
        h, w = img.shape[:2]
        scale = final_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, final_height))
        resized_images.append(resized)
        print(f"Image {idx} resized to: {resized.shape}")

    # Calculate the total width for the final image (panorama)
    total_width = sum(img.shape[1] for img in resized_images)
    print("Total width for movie strip:", total_width)

    # Create a blank canvas with the specified height and calculated total width
    movie_strip = np.zeros((final_height, total_width, 3), dtype=np.uint8)

    # Place each image side by side with no gap
    current_x = 0
    for idx, img in enumerate(resized_images):
        w = img.shape[1]
        movie_strip[:, current_x:current_x+w] = img
        print(f"Placed image {idx} at x = {current_x}")
        current_x += w

    # Save the final combined image
    success = cv2.imwrite(output_file, movie_strip)
    if success:
        print(f"Movie strip saved as {output_file}")
    else:
        print("Failed to save movie strip.")
    return movie_strip

if __name__ == "__main__":
    # Print current working directory to confirm file location
    print("Current working directory:", os.getcwd())
    
    # You can change the final_height value here if needed.
    result = create_movie_strip(final_height=800)
    if result is not None:
        # Create a resizable window
        cv2.namedWindow("Movie Strip", cv2.WINDOW_NORMAL)
        cv2.imshow("Movie Strip", result)
        print("Displaying movie strip. Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
