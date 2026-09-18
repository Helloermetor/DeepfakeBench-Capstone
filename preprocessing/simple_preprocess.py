import cv2
import dlib
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Capstone\DeepfakeBench\datasets\rgb\FaceForensics++")

MODEL = r"C:\Capstone\DeepfakeBench\preprocessing\dlib_tools\shape_predictor_81_face_landmarks.dat"

NUM_FRAMES = 32
IMG_SIZE = 256


def process_video(video_path, output_dir):
    print(f"\nProcessing: {video_path.name}", flush=True)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open video", flush=True)
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        print("Invalid frame count", flush=True)
        cap.release()
        return

    indices = np.linspace(
        0, total - 1, NUM_FRAMES, dtype=int
    )

    saved = 0

    out = output_dir / video_path.stem
    out.mkdir(parents=True, exist_ok=True)

    for n, frame_index in enumerate(indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        ok, frame = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = detector(rgb, 1)

        if len(faces) == 0:
            print(f"Frame {n}: no face", flush=True)
            continue

        # IMPORTANT: use exactly faces[0]
        face = faces[0]

        shape = predictor(rgb, face)

        # Simple bounding-box crop
        x1 = max(0, face.left())
        y1 = max(0, face.top())
        x2 = min(frame.shape[1], face.right())
        y2 = min(frame.shape[0], face.bottom())

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.resize(
            crop,
            (IMG_SIZE, IMG_SIZE)
        )

        filename = out / f"{n:03d}.png"

        cv2.imwrite(str(filename), crop)

        saved += 1

        print(
            f"Frame {n}: saved",
            flush=True
        )

    cap.release()

    print(
        f"Finished {video_path.name}: {saved}/{NUM_FRAMES}",
        flush=True
    )


def main():

    datasets = [
        ROOT / "original_sequences" / "youtube" / "c23" / "videos",
        ROOT / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    ]

    output_root = ROOT / "simple_frames"

    for video_dir in datasets:

        if not video_dir.exists():
            print("Missing:", video_dir)
            continue

        videos = sorted(video_dir.glob("*.mp4"))

        print(
            f"\nFound {len(videos)} videos in {video_dir}",
            flush=True
        )

        for video in videos:
            process_video(video, output_root / video_dir.parent.parent.name)


if __name__ == "__main__":
    main()