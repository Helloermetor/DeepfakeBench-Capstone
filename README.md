# DeepfakeBench Detection

## Overview

> An Xception model initialized with ImageNet-pretrained weights was fine-tuned on FaceForensics++ C23, and the best trained checkpoint (`xception_custom_best.pth`) was used for evaluation.

## Model and Weights

### Model

The project uses the **Xception** architecture provided by DeepfakeBench for binary deepfake classification.

The model classifies each processed face image into two classes:

- `0` — Real
- `1` — Deepfake

### Model Initialization

The custom Xception model was initialized using the ImageNet-pretrained Xception backbone:

```
training/pretrained/xception-b5690688.pth
```

These pretrained weights were used as the starting point for our training.

The model was then fine-tuned on the FaceForensics++ C23 dataset using the custom training script:

```
train_xception_custom.py
```

### Custom Training Checkpoint

The best-performing checkpoint from our custom training was saved as:

```
custom_training/xception_custom_best.pth
```

This is the main trained model checkpoint produced by this project.

It was used for the final evaluation on the held-out FaceForensics++ C23 test set and is the checkpoint intended for the subsequent Celeb-DF v2 cross-dataset evaluation.

### DeepfakeBench Released Weights

DeepfakeBench also contains the following Xception detector checkpoint:

```
training/weights/xception_best.pth
```

This is a released DeepfakeBench detector checkpoint and was not used for our custom training experiment.

### Weight Summary

| File                                        | Description                                   | Role in Our Experiment      |
| ------------------------------------------- | --------------------------------------------- | --------------------------- |
| `training/pretrained/xception-b5690688.pth` | ImageNet-pretrained Xception backbone         | Used to initialize training |
| `custom_training/xception_custom_best.pth`  | Best checkpoint generated during our training | Used for final testing      |
| `training/weights/xception_best.pth`        | Released DeepfakeBench Xception detector      | Not used                    |

### Training Flow

```
ImageNet-pretrained Xception
        ↓
Xception initialization
        ↓
Fine-tuning on FaceForensics++ C23
        ↓
Validation during training
        ↓
Best validation checkpoint
        ↓
xception_custom_best.pth
        ↓
Testing on held-out FF++ C23 test set
```

## Repository Structure

```
DeepfakeBench-Capstone/
│
├── analysis/
├── custom_training/
│   └── xception_custom_best.pth
│
├── datasets/
│   └── rgb/
│       └── FaceForensics++/
│           └── simple_frames/
│               └── dataset.csv
│
├── figures/
├── preprocessing/
│   └── dlib_tools/
│       └── shape_predictor_81_face_landmarks.dat
│
├── results/
├── training/
│   ├── pretrained/
│   │   └── xception-b5690688.pth
│   └── weights/
│       └── xception_best.pth
│
├── create_dataset_csv.py
├── train_xception_custom.py
├── test_custom_xception.py
├── requirements.txt
├── environment.yml
├── README.md
└── ...
```
