from pathlib import Path
import csv
import random

ROOT = Path(
    r"C:\Capstone\DeepfakeBench\datasets\rgb\FaceForensics++\simple_frames"
)

random.seed(42)

rows = []

# REAL = 0
real_root = ROOT / "youtube"

for video_dir in sorted(real_root.iterdir()):
    if not video_dir.is_dir():
        continue

    images = sorted(video_dir.glob("*.png"))

    for image in images:
        rows.append({
            "video_id": video_dir.name,
            "image_path": str(image),
            "label": 0
        })

# FAKE = 1
fake_root = ROOT / "Deepfakes"

for video_dir in sorted(fake_root.iterdir()):
    if not video_dir.is_dir():
        continue

    images = sorted(video_dir.glob("*.png"))

    for image in images:
        rows.append({
            "video_id": video_dir.name,
            "image_path": str(image),
            "label": 1
        })

# Get unique videos separately by class
real_videos = sorted(
    {r["video_id"] for r in rows if r["label"] == 0}
)

fake_videos = sorted(
    {r["video_id"] for r in rows if r["label"] == 1}
)

random.shuffle(real_videos)
random.shuffle(fake_videos)

# 70% train, 10% validation, 20% test
def split_videos(videos):
    n = len(videos)

    train_end = int(0.70 * n)
    val_end = int(0.80 * n)

    return (
        videos[:train_end],
        videos[train_end:val_end],
        videos[val_end:]
    )

real_train, real_val, real_test = split_videos(real_videos)
fake_train, fake_val, fake_test = split_videos(fake_videos)

train_videos = set(real_train + fake_train)
val_videos = set(real_val + fake_val)
test_videos = set(real_test + fake_test)

# Assign split
for r in rows:
    if r["video_id"] in train_videos:
        r["split"] = "train"
    elif r["video_id"] in val_videos:
        r["split"] = "val"
    else:
        r["split"] = "test"

output = ROOT / "dataset.csv"

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image_path", "video_id", "label", "split"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Dataset CSV created:")
print(output)

print("\nVideos:")
print("Train:", len(train_videos))
print("Validation:", len(val_videos))
print("Test:", len(test_videos))

print("\nImages:")
print("Train:", sum(r["split"] == "train" for r in rows))
print("Validation:", sum(r["split"] == "val" for r in rows))
print("Test:", sum(r["split"] == "test" for r in rows))