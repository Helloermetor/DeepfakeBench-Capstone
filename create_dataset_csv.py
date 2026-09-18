import csv
from pathlib import Path
import random

ROOT = Path(r"C:\Capstone\DeepfakeBench")

REAL_ROOT = (
    ROOT
    / "datasets"
    / "rgb"
    / "FaceForensics++"
    / "original_sequences"
    / "youtube"
    / "c23"
    / "frames"
)

FAKE_ROOT = (
    ROOT
    / "datasets"
    / "rgb"
    / "FaceForensics++"
    / "manipulated_sequences"
    / "Deepfakes"
    / "c23"
    / "frames"
)

OUTPUT_CSV = (
    ROOT / "datasets" / "rgb" / "FaceForensics++" / "simple_frames" / "dataset.csv"
)

random.seed(42)

videos = []

# -----------------------------
# REAL VIDEOS
# -----------------------------
for video_dir in sorted(REAL_ROOT.iterdir()):
    if video_dir.is_dir():
        frames = sorted(video_dir.glob("*.png"))

        if frames:
            videos.append({"video_id": video_dir.name, "label": 0, "frames": frames})

# -----------------------------
# DEEPFAKE VIDEOS
# -----------------------------
for video_dir in sorted(FAKE_ROOT.iterdir()):
    if video_dir.is_dir():
        frames = sorted(video_dir.glob("*.png"))

        if frames:
            videos.append({"video_id": video_dir.name, "label": 1, "frames": frames})

print("Real videos found:", sum(v["label"] == 0 for v in videos))
print("Deepfake videos found:", sum(v["label"] == 1 for v in videos))
print("Total videos found:", len(videos))

if len(videos) == 0:
    raise RuntimeError("No videos found. Check the frame folders.")

# -----------------------------
# SHUFFLE VIDEOS
# -----------------------------
random.shuffle(videos)

# 70% train
# 10% validation
# 20% test

total = len(videos)

train_end = int(total * 0.70)
val_end = int(total * 0.80)

for i, video in enumerate(videos):

    if i < train_end:
        video["split"] = "train"

    elif i < val_end:
        video["split"] = "val"

    else:
        video["split"] = "test"

# -----------------------------
# CREATE CSV ROWS
# -----------------------------
rows = []

for video in videos:

    for frame in video["frames"]:

        rows.append(
            {
                "image_path": str(frame),
                "video_id": video["video_id"],
                "label": video["label"],
                "split": video["split"],
            }
        )

# -----------------------------
# SAVE CSV
# -----------------------------
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(f, fieldnames=["image_path", "video_id", "label", "split"])

    writer.writeheader()
    writer.writerows(rows)

print("\nDataset CSV created successfully.")
print("CSV:", OUTPUT_CSV)
print("Total frames:", len(rows))

# -----------------------------
# SPLIT INFORMATION
# -----------------------------
for split in ["train", "val", "test"]:

    split_rows = [r for r in rows if r["split"] == split]

    split_videos = len(set(r["video_id"] for r in split_rows))

    print(split, "videos:", split_videos, "frames:", len(split_rows))
