import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

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

# 🔹 Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (frame_width * 2, frame_height))

# 🔹 Start capturing frames
while cap.isOpened():
    ret, raw_image = cap.read()
    if not ret:
        break

    # 🔹 Resize frame to 640x480
    raw_image = cv2.resize(raw_image, (frame_width, frame_height))

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
