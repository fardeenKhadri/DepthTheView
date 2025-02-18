import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import pygame  # 🔹 Import pygame for sound

# 🔹 Force CPU execution and disable xFormers
import os
os.environ["XFORMERS_FORCE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend=cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 🔹 Set PyTorch to CPU mode
DEVICE = "cpu"
torch.set_default_device(DEVICE)

# 🔹 Load Depth Anything model (Force CPU)
encoder = "vits"
depth_anything = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{encoder}14").to(DEVICE)
depth_anything.eval()

# 🔹 Initialize pygame mixer for sound
pygame.mixer.init()

# 🔹 Load sound file
wav_path = "F:/MAnAsAA/Depth-Anything-main/brou.wav"
if os.path.exists(wav_path):
    pygame.mixer.music.load(wav_path)
else:
    print("❌ WAV file not found!")

# 🔹 Transformation pipeline
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# 🔹 Initialize webcam
cap = cv2.VideoCapture(0)  # 0 = Default Webcam

# 🔹 Get webcam resolution
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 🔹 Define detection box (centered in webcam view)
BOX_WIDTH, BOX_HEIGHT = 200, 200  # Size of detection box
BOX_X1 = (frame_width // 2) - (BOX_WIDTH // 2)
BOX_Y1 = (frame_height // 2) - (BOX_HEIGHT // 2)
BOX_X2 = BOX_X1 + BOX_WIDTH
BOX_Y2 = BOX_Y1 + BOX_HEIGHT

# 🔹 Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (frame_width * 2, frame_height))

# 🔹 Set Depth Threshold (e.g., 3 meters)
THRESHOLD_DISTANCE_METERS = 3.0  # 🚨 Alert only if obstacle is within 3m
DEPTH_SCALE_FACTOR = 100  # 🔹 Adjust based on real-world testing!

# 🔹 Function to detect obstacles within the box & depth threshold
def detect_obstacle_in_box(depth_map, depth_color):
    """Detects if an obstacle is inside the box & within 3m"""
    
    # 🔹 Convert depth map to meters (approximate scaling)
    depth_in_meters = depth_map / DEPTH_SCALE_FACTOR  # Normalize depth

    # 🔹 Detect gold-colored objects
    lower_gold = np.array([20, 180, 180])  # 🔹 Lower bound for gold in HSV
    upper_gold = np.array([40, 255, 255])  # 🔹 Upper bound for gold in HSV
    
    hsv_image = cv2.cvtColor(depth_color, cv2.COLOR_BGR2HSV)  # Convert to HSV
    mask = cv2.inRange(hsv_image, lower_gold, upper_gold)  # Mask for gold objects

    # 🔹 Show gold detection mask (Debugging Purpose)
    cv2.imshow("Gold Mask", mask)

    # 🔹 Get coordinates of gold regions
    gold_pixels = np.where(mask > 0)

    if len(gold_pixels[0]) > 0:
        # 🔹 Filter out pixels inside the detection box
        for i in range(len(gold_pixels[0])):
            y, x = gold_pixels[0][i], gold_pixels[1][i]
            if BOX_X1 <= x <= BOX_X2 and BOX_Y1 <= y <= BOX_Y2:
                # 🔹 Get depth value of the detected gold object
                detected_depth = depth_in_meters[y, x]

                # 🔹 Print detected depth values for debugging
                print(f"Gold Object Detected at ({x},{y}) → Depth: {detected_depth:.2f}m")

                # 🔹 Play sound only if depth is within threshold
                if detected_depth < THRESHOLD_DISTANCE_METERS:
                    return True  # 🚨 Obstacle detected inside the box & close

    return False  # ✅ No close obstacle in the box

# 🔹 Start capturing frames
while cap.isOpened():
    ret, raw_image = cap.read()
    if not ret:
        break

    # 🔹 Resize frame to webcam resolution
    raw_image = cv2.resize(raw_image, (frame_width, frame_height))

    # 🔹 Draw detection box on the main webcam view
    cv2.rectangle(raw_image, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), (0, 255, 0), 2)  # Green Box

    # 🔹 Convert to RGB and normalize
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]

    # 🔹 Apply transformation
    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    # 🔹 Generate Depth Map
    with torch.no_grad():
        depth = depth_anything(image)

    # 🔹 Resize depth map back to original size
    depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)

    # 🔹 Apply color map
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    # 🔹 Check for obstacles inside the box & within 3m
    if detect_obstacle_in_box(depth, depth_color):
        print("🚨 Close Obstacle Inside Box! Playing alert sound...")
        pygame.mixer.music.play()  # 🔹 Play sound

    # 🔹 Combine original frame and depth map side by side
    combined_frame = np.hstack((raw_image, depth_color))

    # 🔹 Write the frame to video file
    out_video.write(combined_frame)

    # 🔹 Show the output
    cv2.imshow("Depth Anything - Webcam", combined_frame)

    # 🔹 Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 🔹 Release resources
cap.release()
out_video.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  # 🔹 Quit pygame mixer