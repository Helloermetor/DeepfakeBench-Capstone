import cv2
import dlib
import numpy as np
from pathlib import Path
from skimage import transform as trans

VIDEO = Path(
    r"C:\Capstone\DeepfakeBench\datasets\rgb\FaceForensics++\original_sequences\youtube\c23\videos\183.mp4"
)

MODEL = Path(
    r"C:\Capstone\DeepfakeBench\preprocessing\dlib_tools\shape_predictor_81_face_landmarks.dat"
)

OUT_DIR = Path(r"C:\Capstone\DeepfakeBench\debug_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("1. Initializing...", flush=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(MODEL))

print("2. Opening video...", flush=True)

cap = cv2.VideoCapture(str(VIDEO))

print("3. Video opened:", cap.isOpened(), flush=True)

ok, frame = cap.read()

print("4. Frame read:", ok, flush=True)

if not ok:
    raise RuntimeError("Could not read frame")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

print("5. First face detection...", flush=True)

faces = detector(rgb, 1)

print("6. Faces:", len(faces), flush=True)

face = max(faces, key=lambda r: r.width() * r.height())

print("7. First landmarks...", flush=True)

shape = predictor(rgb, face)

print("8. Landmarks:", shape.num_parts, flush=True)

points = np.array([
    [shape.part(37).x, shape.part(37).y],
    [shape.part(44).x, shape.part(44).y],
    [shape.part(30).x, shape.part(30).y],
    [shape.part(49).x, shape.part(49).y],
    [shape.part(55).x, shape.part(55).y]
], dtype=np.float32)

print("9. Alignment...", flush=True)

dst = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)

dst[:, 0] += 8.0
dst *= 256.0 / 112.0

tform = trans.SimilarityTransform()

if not tform.estimate(points, dst):
    raise RuntimeError("Alignment failed")

crop = cv2.warpAffine(
    rgb,
    tform.params[:2, :],
    (256, 256)
)

print("10. Crop:", crop.shape, flush=True)

crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

print("11. Second face detection...", flush=True)

faces2 = detector(crop_bgr, 1)

print("12. Second faces:", len(faces2), flush=True)

if len(faces2) == 0:
    raise RuntimeError("No face found after alignment")

print("13. Final landmarks...", flush=True)

shape2 = predictor(crop_bgr, faces2[0])

print("14. Final landmarks:", shape2.num_parts, flush=True)

landmarks = np.array(
    [[p.x, p.y] for p in shape2.parts()],
    dtype=np.float32
)

image_path = OUT_DIR / "frame_000.png"
landmark_path = OUT_DIR / "frame_000.npy"

print("15. Saving image...", flush=True)

success = cv2.imwrite(str(image_path), crop_bgr)

print("16. Image saved:", success, flush=True)

print("17. Saving landmarks...", flush=True)

np.save(str(landmark_path), landmarks)

print("18. Landmarks saved", flush=True)

cap.release()

print("19. COMPLETE", flush=True)