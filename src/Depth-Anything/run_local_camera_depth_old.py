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

# Initialize the model and move it to the desired device
depth_anything = DepthAnything(model_configs[encoder])

# Load the state dictionary with map_location to ensure it's on the correct device
state_dict_path = f'/home/devuser/src/Depth-Anything/checkpoints/depth_anything_{encoder}14.pth'
depth_anything.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))

# Move the model to the device
depth_anything.to(DEVICE)

# Optionally, if you're using the `from_pretrained` method:
# depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE)

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
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'x264' might also be available
out_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (640, 480))

cap = cv2.VideoCapture(video_path)

# Define red color range in HSV
# Red has two ranges in HSV
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

    # Convert to RGB and normalize
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    h, w = image_rgb.shape[:2]

    # Apply transformations for depth model
    image_transformed = transform({'image': image_rgb})['image']
    image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = depth_anything(image_tensor)

    # Resize depth map to original image size
    depth_resized = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0

    depth_uint8 = depth_normalized.cpu().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    # Create a split region for visualization
    split_region = np.ones((h, margin_width, 3), dtype=np.uint8) * 255
    combined_results = cv2.hconcat([raw_image, split_region, depth_color])

    # Create space for captions
    caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
    captions = ['Raw Image', 'Depth Map']
    segment_width = w + margin_width
    for i, caption in enumerate(captions):
        # Calculate text size
        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

        # Calculate x-coordinate to center the text
        text_x = int((segment_width * i) + (w - text_size[0]) / 2)

        # Add text caption
        cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)

    final_result = cv2.vconcat([caption_space, combined_results])

    # =======================
    # Red Object Detection
    # =======================

    # Convert raw image to HSV for color detection
    hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

    # Create masks for red color
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to reduce noise
    # Increase kernel size if necessary for better noise reduction
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for single object detection
    detected = False

    if contours:
        # Select the largest contour assuming it's the primary object
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Define a higher area threshold to filter out smaller noise
        if area > 1000:  # Adjust this threshold based on your specific use case
            detected = True
            # Compute the minimum enclosing circle
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])

                # Ensure the center coordinates are within image bounds
                center_x = np.clip(center_x, 0, w - 1)
                center_y = np.clip(center_y, 0, h - 1)

                # Get depth value at the center
                z_normalized = depth_normalized[int(center_y), int(center_x)]  # Depth value between 0-255

                # Invert Z value to correct orientation
                z_corrected = 255 - z_normalized  # Now, approaching objects have decreasing Z

                # Optionally, map z_corrected to actual depth units if calibration is available
                # For example:
                # z_actual = z_corrected * depth_scale_factor

                # Draw the circle and center on the final_result image
                cv2.circle(final_result, (center_x, center_y), int(radius), (0, 255, 0), 2)
                cv2.circle(final_result, (center_x, center_y), 5, (0, 0, 255), -1)

                # Prepare the text with coordinates
                text = f"X: {center_x}, Y: {center_y}, Z: {z_corrected:.2f}"
                cv2.putText(final_result, text, (center_x + 10, center_y + 10),
                            font, 0.5, (0, 255, 0), 1)

                # Print the coordinates to the terminal
                print(f"Detected Red Object - Center: ({center_x}, {center_y}), Depth (Z): {z_corrected:.2f}")

    # =======================

    # Write the annotated frame to the output video
    # If you want to save the annotated frames, write 'final_result'
    out_video.write(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))  # Convert back to BGR for correct color in video

    # Display the final result
    cv2.imshow('Depth Anything with Red Object Detection', final_result)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()
