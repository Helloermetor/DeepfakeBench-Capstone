import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
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
# CONFIGURATION
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

PRETRAINED_BACKBONE = (
    ROOT
    / "training"
    / "pretrained"
    / "xception-b5690688.pth"
)

OUTPUT_DIR = ROOT / "custom_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = OUTPUT_DIR / "xception_custom_best.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10

LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.0005

THRESHOLD = 0.5
SEED = 1024


# ============================================================
# SEED / DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class FaceDataset(Dataset):

    def __init__(self, rows, transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):

        row = self.rows[index]

        image_path = Path(row["image_path"])
        label = int(row["label"])
        video_id = row["video_id"]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, video_id


# ============================================================
# LOAD CSV
# ============================================================

def load_rows():

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{CSV_FILE}"
        )

    rows = []

    with CSV_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "image_path",
            "video_id",
            "label",
            "split",
        }

        if not required_columns.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "CSV must contain columns: "
                "image_path, video_id, label, split"
            )

        for row in reader:

            image_path = Path(row["image_path"])

            if not image_path.exists():

                print(
                    "WARNING - missing image:",
                    image_path
                )

                continue

            rows.append(row)

    return rows


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


# ============================================================
# BUILD XCEPTION
# ============================================================

def build_model():

    config = {
        "mode": "original",
        "num_classes": 2,
        "inc": 3,
        "dropout": False,
    }

    print(
        "Creating Xception...",
        flush=True
    )

    model = Xception(config)

    if not PRETRAINED_BACKBONE.exists():

        raise FileNotFoundError(
            "Pretrained Xception backbone not found:\n"
            f"{PRETRAINED_BACKBONE}"
        )

    print(
        "Loading pretrained Xception backbone...",
        flush=True
    )

    state_dict = torch.load(
        str(PRETRAINED_BACKBONE),
        map_location="cpu"
    )

    # Convert pointwise convolution weights
    # from 2D to 4D when needed.
    for name, value in list(
        state_dict.items()
    ):

        if (
            "pointwise" in name
            and hasattr(value, "ndim")
            and value.ndim == 2
        ):

            state_dict[name] = (
                value
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

    # Remove ImageNet classifier weights.
    state_dict = {
        name: value
        for name, value in state_dict.items()
        if "fc" not in name
    }

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False
        )
    )

    print(
        "Backbone loaded.",
        flush=True
    )

    print(
        f"Missing keys: {len(missing)}",
        flush=True
    )

    print(
        f"Unexpected keys: {len(unexpected)}",
        flush=True
    )

    return model


# ============================================================
# EXTRACT LOGITS
# ============================================================

def get_logits(model_output):

    """
    DeepfakeBench Xception may return:

        (logits, features)

    CrossEntropyLoss needs only the logits tensor.
    """

    if isinstance(
        model_output,
        tuple
    ):

        logits = model_output[0]

    else:

        logits = model_output

    if not torch.is_tensor(logits):

        raise TypeError(
            "Expected model output to contain "
            f"a Tensor, got {type(logits)}"
        )

    return logits


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_probs = []

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

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            output = model(images)

            logits = get_logits(
                output
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            labels_cpu = (
                labels
                .detach()
                .cpu()
                .numpy()
            )

            probs_cpu = (
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels_cpu.tolist()
            )

            all_probs.extend(
                probs_cpu.tolist()
            )

            for (
                video_id,
                probability,
                label
            ) in zip(
                video_ids,
                probs_cpu,
                labels_cpu
            ):

                video_probs[
                    video_id
                ].append(
                    float(probability)
                )

                video_labels[
                    video_id
                ] = int(label)

    if len(all_labels) == 0:

        raise RuntimeError(
            "No samples found during evaluation."
        )

    # --------------------------------------------------------
    # FRAME LEVEL
    # --------------------------------------------------------

    frame_predictions = [
        1 if p >= THRESHOLD else 0
        for p in all_probs
    ]

    precision = precision_score(
        all_labels,
        frame_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        frame_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        frame_predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            all_labels,
            all_probs
        )

    except ValueError:

        auc = float("nan")

    cm = confusion_matrix(
        all_labels,
        frame_predictions,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # VIDEO LEVEL
    # --------------------------------------------------------

    video_labels_list = []
    video_scores_list = []

    for video_id in sorted(
        video_probs
    ):

        mean_score = float(
            np.mean(
                video_probs[video_id]
            )
        )

        video_scores_list.append(
            mean_score
        )

        video_labels_list.append(
            video_labels[video_id]
        )

    video_predictions = [
        1 if score >= THRESHOLD else 0
        for score in video_scores_list
    ]

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

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,

        "video_precision": video_precision,
        "video_recall": video_recall,
        "video_f1": video_f1,
        "video_auc": video_auc,

        "confusion_matrix": cm,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CUSTOM XCEPTION DEEPFAKE DETECTOR")
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
        "CSV:",
        CSV_FILE
    )

    print(
        "Backbone:",
        PRETRAINED_BACKBONE
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    rows = load_rows()

    train_rows = [
        row
        for row in rows
        if row["split"] == "train"
    ]

    val_rows = [
        row
        for row in rows
        if row["split"] == "val"
    ]

    test_rows = [
        row
        for row in rows
        if row["split"] == "test"
    ]

    print("\nDataset:")
    print(
        "Total:",
        len(rows)
    )

    print(
        "Train:",
        len(train_rows)
    )

    print(
        "Validation:",
        len(val_rows)
    )

    print(
        "Test:",
        len(test_rows)
    )

    if not train_rows:
        raise RuntimeError(
            "Training dataset is empty."
        )

    if not val_rows:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    if not test_rows:
        raise RuntimeError(
            "Test dataset is empty."
        )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = FaceDataset(
        train_rows,
        train_transform
    )

    val_dataset = FaceDataset(
        val_rows,
        eval_transform
    )

    test_dataset = FaceDataset(
        test_rows,
        eval_transform
    )

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    test_loader = DataLoader(
        test_dataset,
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

    model = build_model()

    model = model.to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_auc = -1.0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for (
            images,
            labels,
            _
        ) in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            output = model(
                images
            )

            logits = get_logits(
                output
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = (
            running_loss
            / total
        )

        train_accuracy = (
            correct
            / total
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_results = evaluate(
            model,
            val_loader
        )

        print(
            "\n" + "-" * 50
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss:     "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Precision:  "
            f"{val_results['precision']:.4f}"
        )

        print(
            f"Val Recall:     "
            f"{val_results['recall']:.4f}"
        )

        print(
            f"Val F1:         "
            f"{val_results['f1']:.4f}"
        )

        print(
            f"Val AUROC:      "
            f"{val_results['auc']:.4f}"
        )

        current_auc = (
            val_results["auc"]
        )

        if (
            not np.isnan(current_auc)
            and current_auc > best_val_auc
        ):

            best_val_auc = (
                current_auc
            )

            torch.save(
                model.state_dict(),
                BEST_MODEL
            )

            print(
                "\nBest model saved:"
            )

            print(
                BEST_MODEL
            )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL TEST"
    )

    print(
        "=" * 60
    )

    if BEST_MODEL.exists():

        best_state = torch.load(
            str(BEST_MODEL),
            map_location=DEVICE
        )

        model.load_state_dict(
            best_state
        )

    test_results = evaluate(
        model,
        test_loader
    )

    print(
        "\nFrame-level metrics:"
    )

    print(
        f"Precision: "
        f"{test_results['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{test_results['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{test_results['f1']:.4f}"
    )

    print(
        f"AUROC:     "
        f"{test_results['auc']:.4f}"
    )

    print(
        "\nVideo-level metrics:"
    )

    print(
        f"Precision: "
        f"{test_results['video_precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{test_results['video_recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{test_results['video_f1']:.4f}"
    )

    print(
        f"AUROC:     "
        f"{test_results['video_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        test_results["confusion_matrix"]
    )

    print(
        "\nTraining complete."
    )


if __name__ == "__main__":
    main()