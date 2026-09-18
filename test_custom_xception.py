import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from training.networks.xception import Xception


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Capstone\DeepfakeBench")

CSV_FILE = (
    ROOT
    / "datasets"
    / "rgb"
    / "FaceForensics++"
    / "simple_frames"
    / "dataset.csv"
)

MODEL_FILE = (
    ROOT
    / "custom_training"
    / "xception_custom_best.pth"
)

IMAGE_SIZE = 256
BATCH_SIZE = 16
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class FaceDataset(Dataset):

    def __init__(self, rows):

        self.rows = rows

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):

        row = self.rows[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        image = self.transform(image)

        label = int(row["label"])

        video_id = row["video_id"]

        return image, label, video_id


# ============================================================
# LOAD CSV
# ============================================================

def load_test_rows():

    rows = []

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["split"] != "test":
                continue

            if not Path(
                row["image_path"]
            ).exists():

                continue

            rows.append(row)

    return rows


# ============================================================
# MODEL
# ============================================================

def build_model():

    config = {
        "mode": "original",
        "num_classes": 2,
        "inc": 3,
        "dropout": False,
    }

    model = Xception(config)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# GET LOGITS
# ============================================================

def get_logits(output):

    if isinstance(output, tuple):

        return output[0]

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CUSTOM XCEPTION TESTING")
    print("=" * 60)

    print(
        "Device:",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Model:",
        MODEL_FILE
    )

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    rows = load_test_rows()

    print(
        "\nTest frames:",
        len(rows)
    )

    if len(rows) == 0:

        raise RuntimeError(
            "No test frames found."
        )

    dataset = FaceDataset(rows)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading Xception..."
    )

    model = build_model()

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    frame_labels = []
    frame_probs = []

    video_probs = defaultdict(list)
    video_labels = {}

    with torch.no_grad():

        for (
            images,
            labels,
            video_ids
        ) in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            output = model(images)

            logits = get_logits(output)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            probabilities = (
                probabilities
                .cpu()
                .numpy()
            )

            labels = (
                labels
                .numpy()
            )

            frame_labels.extend(
                labels.tolist()
            )

            frame_probs.extend(
                probabilities.tolist()
            )

            for (
                video_id,
                probability,
                label
            ) in zip(
                video_ids,
                probabilities,
                labels
            ):

                video_probs[
                    video_id
                ].append(
                    float(probability)
                )

                video_labels[
                    video_id
                ] = int(label)

    # ========================================================
    # FRAME LEVEL
    # ========================================================

    frame_predictions = [
        1 if p >= THRESHOLD else 0
        for p in frame_probs
    ]

    frame_accuracy = accuracy_score(
        frame_labels,
        frame_predictions
    )

    frame_precision = precision_score(
        frame_labels,
        frame_predictions,
        zero_division=0
    )

    frame_recall = recall_score(
        frame_labels,
        frame_predictions,
        zero_division=0
    )

    frame_f1 = f1_score(
        frame_labels,
        frame_predictions,
        zero_division=0
    )

    try:

        frame_auc = roc_auc_score(
            frame_labels,
            frame_probs
        )

    except ValueError:

        frame_auc = float("nan")

    frame_cm = confusion_matrix(
        frame_labels,
        frame_predictions,
        labels=[0, 1]
    )

    # ========================================================
    # VIDEO LEVEL
    # ========================================================

    video_labels_list = []
    video_scores_list = []

    for video_id in sorted(
        video_probs
    ):

        score = float(
            np.mean(
                video_probs[video_id]
            )
        )

        video_scores_list.append(
            score
        )

        video_labels_list.append(
            video_labels[video_id]
        )

    video_predictions = [
        1 if p >= THRESHOLD else 0
        for p in video_scores_list
    ]

    video_accuracy = accuracy_score(
        video_labels_list,
        video_predictions
    )

    video_precision = precision_score(
        video_labels_list,
        video_predictions,
        zero_division=0
    )

    video_recall = recall_score(
        video_labels_list,
        video_predictions,
        zero_division=0
    )

    video_f1 = f1_score(
        video_labels_list,
        video_predictions,
        zero_division=0
    )

    try:

        video_auc = roc_auc_score(
            video_labels_list,
            video_scores_list
        )

    except ValueError:

        video_auc = float("nan")

    video_cm = confusion_matrix(
        video_labels_list,
        video_predictions,
        labels=[0, 1]
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("FRAME-LEVEL RESULTS")
    print("=" * 60)

    print(
        f"Accuracy:  {frame_accuracy:.4f}"
    )

    print(
        f"Precision: {frame_precision:.4f}"
    )

    print(
        f"Recall:    {frame_recall:.4f}"
    )

    print(
        f"F1:        {frame_f1:.4f}"
    )

    print(
        f"AUROC:     {frame_auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(frame_cm)

    print("\n" + "=" * 60)
    print("VIDEO-LEVEL RESULTS")
    print("=" * 60)

    print(
        f"Videos tested: {len(video_labels_list)}"
    )

    print(
        f"Accuracy:  {video_accuracy:.4f}"
    )

    print(
        f"Precision: {video_precision:.4f}"
    )

    print(
        f"Recall:    {video_recall:.4f}"
    )

    print(
        f"F1:        {video_f1:.4f}"
    )

    print(
        f"AUROC:     {video_auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(video_cm)

    print(
        "\nTesting complete."
    )


if __name__ == "__main__":
    main()