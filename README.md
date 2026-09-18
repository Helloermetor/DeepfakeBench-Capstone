# Deepfake Detection using Xception and DeepfakeBench

## 1. Project Overview

This project implements a deepfake detection system using the **DeepfakeBench framework** and an **Xception-based deepfake detector**.

The system is designed to classify videos/facial frames into two categories:

- **Real**
- **Deepfake**

The current experiment uses the **FaceForensics++ dataset with C23 compression**.

The project includes the complete workflow from raw videos to face preprocessing, dataset preparation, model training, frame-level testing, video-level prediction and evaluation.

The next stage of the project is **cross-dataset evaluation on Celeb-DF v2** using the existing trained Xception model.

---

# 2. Project Objectives

The main objectives of this project are:

1. Build a deepfake detection pipeline using DeepfakeBench.
2. Prepare real and manipulated videos for deepfake detection.
3. Extract frames from videos.
4. Detect faces using Dlib.
5. Detect facial landmarks.
6. Align and crop faces.
7. Resize face crops to 256 × 256.
8. Create a dataset CSV for model training and testing.
9. Train an Xception-based binary classifier.
10. Evaluate the model at frame level.
11. Aggregate frame predictions to obtain video-level predictions.
12. Calculate Precision, Recall, F1-score and AUROC.
13. Perform cross-dataset evaluation using Celeb-DF v2.
14. Analyze the generalization of the model across datasets.

---

# 3. Complete Project Workflow

The complete project workflow is:

```text
                    DATASET
                       │
                       ▼
              FaceForensics++
                       │
                       ▼
               Video Selection
                       │
                       ▼
                Frame Sampling
                       │
                       ▼
               Face Detection
                    (Dlib)
                       │
                       ▼
            Facial Landmark Detection
                       │
                       ▼
                Face Alignment
                       │
                       ▼
                Face Cropping
                       │
                       ▼
              Resize to 256 × 256
                       │
                       ▼
                Dataset CSV
                       │
                       ▼
              Train / Validation /
                  Test Split
                       │
                       ▼
                 Xception Model
                       │
                       ▼
                   Training
                       │
                       ▼
             Best Model Checkpoint
                       │
                       ▼
                    Testing
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Frame-Level          Video-Level
        Evaluation           Evaluation
             │                   │
             └─────────┬─────────┘
                       ▼
       Precision / Recall / F1 / AUROC
                       │
                       ▼
              Final Analysis
                       │
                       ▼
       Celeb-DF v2 Cross-Dataset
              Evaluation
```
