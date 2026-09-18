import cv2
import dlib
import numpy as np
from pathlib import Path
from skimage import transform as trans

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"C:\Capstone\DeepfakeBench\datasets\rgb\FaceForensics++")

LANDMARK_MODEL = Path(
    r"C:\Capstone\DeepfakeBench\preprocessing\dlib_tools\shape_predictor_81_face_landmarks.dat"
)

NUM_FRAMES = 32
RES = 256


# ============================================================
# LANDMARKS
# ============================================================

def get_five_points(shape):
    return np.array([
        [shape.part(37).x, shape.part(37).y],  # left eye
        [shape.part(44).x, shape.part(44).y],  # right eye
        [shape.part(30).x, shape.part(30).y],  # nose
        [shape.part(49).x, shape.part(49).y],  # left mouth
        [shape.part(55).x, shape.part(55).y],  # right mouth
    ], dtype=np.float32)


# ============================================================
# FACE ALIGNMENT
# ============================================================

def align_face(image, points, size=256):

    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]
    ], dtype=np.float32)

    dst[:, 0] += 8.0
    dst *= size / 112.0

    tform = trans.SimilarityTransform()

    if not tform.estimate(points, dst):
        return None

    matrix = tform.params[:2, :]

    return cv2.warpAffine(
        image,
        matrix,
        (size, size)
    )


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path, detector, predictor):

    print()
    print("=" * 60)
    print(f"Processing: {video_path.name}")
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video")
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Frame count: {frame_count}")

    if frame_count <= 0:
        print("ERROR: Invalid frame count")
        cap.release()
        return 0

    # Select 32 evenly spaced frame numbers
    target_indices = np.linspace(
        0,
        frame_count - 1,
        NUM_FRAMES,
        dtype=int
    )

    target_set = set(target_indices.tolist())

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_root = video_path.parent.parent / "frames"
    output_dir = output_root / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    landmark_root = video_path.parent.parent / "landmarks"
    landmark_dir = landmark_root / video_path.stem
    landmark_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    current_frame = 0

    # --------------------------------------------------------
    # Sequential video reading
    # --------------------------------------------------------

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if current_frame not in target_set:
            current_frame += 1
            continue

        try:

            print(
                f"Frame {current_frame} "
                f"({saved + 1}/{NUM_FRAMES})"
            )

            # OpenCV BGR -> RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # Face detection
            # ------------------------------------------------

            faces = detector(rgb, 1)

            if len(faces) == 0:
                print("  No face detected - skipped")
                current_frame += 1
                continue

            print("  Face detected")

            # ------------------------------------------------
            # Landmarks
            # ------------------------------------------------

            shape = predictor(rgb, faces[0])

            points = get_five_points(shape)

            # ------------------------------------------------
            # Alignment
            # ------------------------------------------------

            cropped_rgb = align_face(
                rgb,
                points,
                RES
            )

            if cropped_rgb is None:
                print("  Alignment failed - skipped")
                current_frame += 1
                continue

            # RGB -> BGR
            cropped_bgr = cv2.cvtColor(
                cropped_rgb,
                cv2.COLOR_RGB2BGR
            )

            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            image_file = (
                output_dir /
                f"{saved:03d}.png"
            )

            landmark_file = (
                landmark_dir /
                f"{saved:03d}.npy"
            )

            success = cv2.imwrite(
                str(image_file),
                cropped_bgr
            )

            if not success:
                print("  ERROR: Could not save image")
                current_frame += 1
                continue

            np.save(
                str(landmark_file),
                points
            )

            saved += 1

            print("  SAVED")

        except Exception as e:

            print(
                f"  ERROR processing frame: {e}"
            )

        current_frame += 1

    cap.release()

    print(
        f"Finished {video_path.name}: "
        f"{saved}/{NUM_FRAMES} frames"
    )

    return saved


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SAFE DEEPFAKEBENCH PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Check landmark model
    # --------------------------------------------------------

    if not LANDMARK_MODEL.exists():

        print()
        print("ERROR: Landmark model not found:")
        print(LANDMARK_MODEL)
        return

    print()
    print("Loading dlib detector...")

    detector = dlib.get_frontal_face_detector()

    print("Loading landmark predictor...")

    predictor = dlib.shape_predictor(
        str(LANDMARK_MODEL)
    )

    print("Dlib initialized successfully")

    # --------------------------------------------------------
    # Dataset directories
    # --------------------------------------------------------

    datasets = [

        ROOT /
        "original_sequences" /
        "youtube" /
        "c23" /
        "videos",

        ROOT /
        "manipulated_sequences" /
        "Deepfakes" /
        "c23" /
        "videos"
    ]

    total_videos = 0
    total_frames = 0

    # --------------------------------------------------------
    # Process datasets
    # --------------------------------------------------------

    for video_dir in datasets:

        print()
        print("=" * 60)
        print(f"Dataset: {video_dir}")
        print("=" * 60)

        if not video_dir.exists():

            print(
                f"WARNING: Directory does not exist: "
                f"{video_dir}"
            )

            continue

        videos = sorted(
            video_dir.glob("*.mp4")
        )

        print(
            f"Found {len(videos)} videos"
        )

        for video in videos:

            try:

                saved = process_video(
                    video,
                    detector,
                    predictor
                )

                total_videos += 1
                total_frames += saved

            except Exception as e:

                print()
                print(
                    f"VIDEO ERROR: {video.name}"
                )
                print(e)

                continue

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Videos processed: {total_videos}"
    )

    print(
        f"Frames saved: {total_frames}"
    )


if __name__ == "__main__":
    main()