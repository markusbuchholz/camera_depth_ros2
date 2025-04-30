import cv2
import numpy as np
import glob

def align_images(base_img, img_to_align, max_features=500, good_match_percent=1.0):
    """
    Aligns img_to_align to base_img using ORB keypoints and homography.
    """
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(descriptors1, descriptors2, None))  # Convert tuple to list
    
    # Sort matches by score (lower distance is better)
    matches.sort(key=lambda x: x.distance)
    
    # Remove not-so-good matches
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]
    
    # Extract locations of good matches.
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        
    # Find homography matrix.
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    # Use homography to warp image.
    height, width, channels = base_img.shape
    aligned_img = cv2.warpPerspective(img_to_align, h, (width, height))
    
    return aligned_img

def create_composite(image_folder="images/", output_file="composite.jpg"):
    # Get list of image files (change pattern if needed)
    image_files = sorted(glob.glob(image_folder + "*.jpg"))
    if not image_files:
        print("No images found in the folder.")
        return

    # Read the base image.
    base_img = cv2.imread(image_files[0])
    if base_img is None:
        print("Failed to load the base image.")
        return
    
    # Create an accumulator for blending (use float32 for precision).
    composite = np.zeros_like(base_img, dtype=np.float32)
    num_images = len(image_files)
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing {image_file} ({idx+1}/{num_images})")
        img = cv2.imread(image_file)
        if img is None:
            print(f"Failed to load {image_file}. Skipping.")
            continue
        # Align every image to the base image.
        if idx != 0:
            img = align_images(base_img, img)
        composite += img.astype(np.float32)
    
    # Average the composite image.
    composite /= num_images
    composite = composite.astype(np.uint8)
    
    # Save the final composite.
    cv2.imwrite(output_file, composite)
    print(f"Composite image saved as {output_file}")

if __name__ == "__main__":
    create_composite()
