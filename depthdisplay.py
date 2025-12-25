import cv2
import torch
import numpy as np

#loading midas from pytorch
model_type = "MiDaS_small"  # smaller version (faster performance)
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# preprocessing transformers
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# cuda gpu/cpu based off of availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)

#camera and stream setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #transforming + converting to batch 
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        # upsampling to original frame resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    # convert depth prediction to numpy + normalization of depth map
    depth_map = prediction.cpu().numpy()
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (255 * ((depth_map - depth_min) / (depth_max - depth_min))).astype(np.uint8)
    #colormap
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

    #side-by-side display
    combined = np.hstack((frame, depth_colored))
    cv2.imshow("Webcam & Depth Map", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
