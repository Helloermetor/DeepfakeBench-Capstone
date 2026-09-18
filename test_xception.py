import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(r"C:\Capstone\DeepfakeBench")

sys.path.insert(0, str(ROOT))

from training.detectors.xception_detector import XceptionDetector


WEIGHTS_PATH = ROOT / "training" / "weights" / "xception_best.pth"

# Find one processed video containing face crops
video_dir = next(
    p for p in
    (ROOT / "datasets" / "rgb" / "FaceForensics++" / "simple_frames" / "Deepfakes").iterdir()
    if p.is_dir()
)

image_paths = sorted(video_dir.glob("*.png"))

if len(image_paths) < 32:
    raise RuntimeError(
        f"Need 32 frames, but only found {len(image_paths)} in {video_dir}"
    )

image_paths = image_paths[:32]

print("Testing video:", video_dir)
print("Number of frames:", len(image_paths))
print("Weights:", WEIGHTS_PATH)

config = {
    "pretrained": str(
        ROOT / "training" / "pretrained" / "xception-b5690688.pth"
    ),
    "model_name": "xception",
    "backbone_name": "xception",

    "backbone_config": {
        "mode": "original",
        "num_classes": 2,
        "inc": 3,
        "dropout": False,
    },

    "loss_func": "cross_entropy",
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Build detector
model = XceptionDetector(config)

# Load released detector checkpoint
checkpoint = torch.load(
    str(WEIGHTS_PATH),
    map_location=device
)

model.load_state_dict(checkpoint, strict=True)

model = model.to(device)
model.eval()

# Same normalization used by DeepfakeBench Xception config
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# Load exactly 32 frames
frames = []

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    frames.append(transform(image))

# Shape = [32, 3, 256, 256]
images = torch.stack(frames, dim=0)

print("Input tensor shape:", images.shape)

images = images.to(device)

# DeepfakeBench Xception expects image tensor shaped like:
# [32, 3, 256, 256]
data_dict = {
    "image": images
}

with torch.no_grad():
    output = model(data_dict)

print("\nRaw output shape:")
print(output["cls"].shape)

print("\nFrame probabilities:")
print(output["prob"].detach().cpu().numpy())

# Average the 32 frame probabilities
video_score = output["prob"].mean().item()

print("\n==========================")
print("VIDEO RESULT")
print("==========================")
print("Deepfake score:", video_score)

if video_score >= 0.5:
    print("Prediction: FAKE")
else:
    print("Prediction: REAL")