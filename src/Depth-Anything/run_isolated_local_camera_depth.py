import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Define the encoder and video source
encoder = 'vits'  # Options: 'vits', 'vitb', 'vitl'
video_path = 0     # 0 for webcam or provide a path to a video file

# Define UI parameters
margin_width = 50
caption_height = 60

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize DepthAnything model
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

depth_anything = DepthAnything(model_configs[encoder])

# Load the state dictionary with map_location
state_dict_path = f'/home/devuser/src/Depth-Anything/checkpoints/depth_anything_{encoder}14.pth'
depth_anything.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))
depth_anything.to(DEVICE)

total_params = sum(param.numel() for param in depth_anything.parameters())
print('Total parameters: {:.2f}M'.format(total_params / 1e6))

depth_anything.eval()

# Define the image transformations
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'x264'
out_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (640, 480))

cap = cv2.VideoCapture(video_path)

# Define red color range in HSV
red_lower1 = np.array([0, 70, 50])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 70, 50])
red_upper2 = np.array([180, 255, 255])

while cap.isOpened():
    ret, raw_image = cap.read()
    if not ret:
        break

    # Resize the raw image
    raw_image = cv2.resize(raw_image, (640, 480))
    # Convert to RGB for color detection or other tasks
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    h, w = image_rgb.shape[:2]

    # =================================================================
    # STEP 1: Detect the red object to find ROI (region of interest)
    # =================================================================
    hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables
    roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, w, h
    detected = False

    if contours:
        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area > 1000:  # Example threshold
            detected = True
            x, y, width_roi, height_roi = cv2.boundingRect(largest_contour)

            # Expand the bounding box slightly to capture more context
            padding = 10
            roi_x1 = max(x - padding, 0)
            roi_y1 = max(y - padding, 0)
            roi_x2 = min(x + width_roi + padding, w)
            roi_y2 = min(y + height_roi + padding, h)

    # =================================================================
    # STEP 2: Crop the region of interest and run depth estimation only
    #         on that cropped region (if detected).
    # =================================================================
    cropped_rgb = image_rgb[roi_y1:roi_y2, roi_x1:roi_x2, :].astype(np.float32) / 255.0
    crop_h, crop_w = cropped_rgb.shape[:2]

    # Apply transforms for the cropped region
    image_transformed = transform({'image': cropped_rgb})['image']
    image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth_cropped = depth_anything(image_tensor)

    # Resize depth map to ROI size
    depth_cropped_resized = F.interpolate(depth_cropped[None], 
                                          (crop_h, crop_w), 
                                          mode='bilinear',
                                          align_corners=False)[0, 0]
    # Normalize for visualization
    min_val = depth_cropped_resized.min()
    max_val = depth_cropped_resized.max()
    if (max_val - min_val) > 1e-6:
        depth_cropped_norm = (depth_cropped_resized - min_val) / (max_val - min_val) * 255.0
    else:
        depth_cropped_norm = depth_cropped_resized * 0

    depth_cropped_uint8 = depth_cropped_norm.cpu().numpy().astype(np.uint8)
    depth_cropped_color = cv2.applyColorMap(depth_cropped_uint8, cv2.COLORMAP_INFERNO)

    # ======================
    # Paste the cropped depth map onto a blank depth image
    # the same size as the original image
    # ======================
    depth_full_color = np.zeros((h, w, 3), dtype=np.uint8)
    depth_full_color[roi_y1:roi_y2, roi_x1:roi_x2, :] = depth_cropped_color

    # =================================================================
    # STEP 3: Visualization
    # =================================================================
    # Create a split region for side-by-side results
    split_region = np.ones((h, margin_width, 3), dtype=np.uint8) * 255
    combined_results = cv2.hconcat([raw_image, split_region, depth_full_color])

    # Create caption space
    caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
    captions = ['Raw Image', 'Depth Map (ROI-based)']
    segment_width = w + margin_width

    for i, caption in enumerate(captions):
        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]
        text_x = int((segment_width * i) + (w - text_size[0]) / 2)
        cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)

    final_result = cv2.vconcat([caption_space, combined_results])

    # ======================
    # STEP 4: If we have a valid red object, visualize its center + depth
    # ======================
    if detected:
        # Recompute the minimum enclosing circle or center
        ((cx, cy), radius) = cv2.minEnclosingCircle(largest_contour)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            # Clip to image boundaries
            center_x = np.clip(center_x, 0, w - 1)
            center_y = np.clip(center_y, 0, h - 1)

            # Depth in the pasted region
            z_normalized = depth_full_color[center_y, center_x, 0]  # just take the first channel or compute average
            z_corrected = 255 - z_normalized

            # Draw circle and center
            cv2.circle(final_result, (center_x, center_y + caption_height), 
                       int(radius), (0, 255, 0), 2)
            cv2.circle(final_result, (center_x, center_y + caption_height), 
                       5, (0, 0, 255), -1)

            # Prepare text
            text = f"X: {center_x}, Y: {center_y}, Z: {z_corrected:.2f}"
            cv2.putText(final_result, text, (center_x + 10, center_y + caption_height + 10), 
                        font, 0.5, (0, 255, 0), 1)

            print(f"Detected Red Object - Center: ({center_x}, {center_y}), Depth (Z): {z_corrected:.2f}")

    # Write the annotated frame to output video (convert BGR <-> RGB if needed)
    out_video.write(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))

    # Display the result
    cv2.imshow('Depth Anything - ROI-based Depth for Red Object', final_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()
