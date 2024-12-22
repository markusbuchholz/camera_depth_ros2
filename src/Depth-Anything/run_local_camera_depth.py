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

# =============================
# NEW: Depth scale factor
# =============================
# You can use this factor to scale the depth. 
# Values > 1 will increase the range; values < 1 will reduce it.
depth_scale_factor = 2.0  # Example: Double the range of Z

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
    depth_resized = F.interpolate(
        depth[None], (h, w),
        mode='bilinear', align_corners=False
    )[0, 0]

    # -----------------------------------------------------------------
    # Approach A) Scale *before* converting to 8-bit for color map:
    # This will also change how your color map is visually stretched.
    # -----------------------------------------------------------------
    # 1. Normalize to [0, 1]
    depth_min, depth_max = depth_resized.min(), depth_resized.max()
    depth_norm_01 = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)

    # 2. Apply your custom scale factor. This can stretch values beyond 1 if > 1.
    depth_scaled = depth_norm_01 * depth_scale_factor  # e.g., multiply by 2

    # 3. Clip to [0, 1] in case scale_factor > 1
    depth_scaled_clipped = torch.clamp(depth_scaled, 0.0, 1.0)

    # 4. Convert to 0-255 for visualization
    depth_uint8 = (depth_scaled_clipped * 255.0).cpu().numpy().astype(np.uint8)

    # Now apply colormap
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    # -----------------------------------------------------------------
    # Approach B) Keep color map as is, only scale the Z value used 
    # for object detection. (See below in the detection part)
    # -----------------------------------------------------------------

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
    hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

    # Create masks for red color
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Higher area threshold to filter out smaller noise
        if area > 1000:
            detected = True
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])

                center_x = np.clip(center_x, 0, w - 1)
                center_y = np.clip(center_y, 0, h - 1)

                # ---------------------------------------------------------------
                # If you kept the color map in the original 0-255 range (Approach B),
                # you can still scale the numeric Z for your own logic below.
                # Remember: 'depth_norm_01' is in [0,1] (pre-scaled).
                # So we could do:
                #   z_normalized = depth_norm_01[center_y, center_x].item() * 255
                # Then "invert" and scale as you wish:
                # ---------------------------------------------------------------
                
                # 1) Get the normalized 0-1 value
                z_norm_01 = depth_norm_01[center_y, center_x].item()
                
                # 2) Convert to 0-255 range
                z_0_255 = z_norm_01 * 255

                # 3) Invert so that closer objects are higher Z 
                z_corrected = 255 - z_0_255

                # 4) Scale the final numeric (for your use/captions)
                z_corrected_scaled = z_corrected * depth_scale_factor
                
                # Draw the circle and center on the final_result image
                cv2.circle(final_result, (center_x, center_y), int(radius), (0, 255, 0), 2)
                cv2.circle(final_result, (center_x, center_y), 5, (0, 0, 255), -1)

                # Prepare the text with coordinates (use z_corrected_scaled if you want the scaled depth)
                text = (f"X: {center_x}, Y: {center_y}, "
                        f"Z: {z_corrected_scaled:.2f}")
                cv2.putText(final_result, text, (center_x + 10, center_y + 10),
                            font, 0.5, (0, 255, 0), 1)

                # Print to terminal
                print(
                    f"Detected Red Object - Center: ({center_x}, {center_y}), "
                    f"Depth (Z scaled): {z_corrected_scaled:.2f}"
                )

    # Write the annotated frame to the output video
    out_video.write(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))

    # Display the final result
    cv2.imshow('Depth Anything with Red Object Detection', final_result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()
