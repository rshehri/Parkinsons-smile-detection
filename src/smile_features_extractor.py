"""
Smile feature extractor for Parkinson's disease detection.

Converted from `notebooks/Smile_Features_Extractor.ipynb` (originally a Google
Colab notebook) into a self-contained, VS Code / command-line friendly script.

Pipeline (per video clip):
  1. Run OpenFace `FeatureExtraction` to obtain Facial Action Unit (AU) values.
  2. Run MediaPipe FaceMesh to obtain 468 3D facial landmarks per frame.
  3. Derive 7 geometric features from the landmarks (eye/eyebrow/mouth/jaw
     openness, mouth width), normalised by the inter-canthal distance (ICD).
  4. Merge the per-frame geometric features with the OpenFace AUs.
  5. Aggregate every clip into a single row of statistics (mean, variance and
     histogram entropy) for each of the 14 signals -> the model-ready CSV
     (same layout as `data/Extracted_feautures2.csv`, minus the `PD` label).

Outputs (written under --output-dir):
  <clip>_landmarks.csv        raw MediaPipe landmarks per frame
  per_frame_features.csv      geometric features + AUs, all frames, all clips
  clip_stats.csv              one aggregated row per clip (feed this to training)

NOTE ON REQUIREMENTS
--------------------
This stage needs external tools that are NOT pip-installable:
  * OpenFace  - build it once and pass the binary via --openface-bin
                https://github.com/TadasBaltrusaitis/OpenFace/wiki
  * mediapipe + opencv-python  (pip install mediapipe opencv-python)
See the README for full setup instructions.

The `PD` label column (0 = healthy, 1 = Parkinson's) is NOT produced here; it
is a clinical ground-truth label that you must join onto `clip_stats.csv`
yourself before training.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import entropy


# --------------------------------------------------------------------------- #
# Configuration: OpenFace Action Units and MediaPipe landmark pairs
# --------------------------------------------------------------------------- #

# Action Units extracted by OpenFace that we keep (regression + presence).
AUS = ["AU01", "AU06", "AU12", "AU14", "AU25", "AU26", "AU45"]
AU_R = [f"{au}_r" for au in AUS]  # intensity (regression) columns
AU_C = [f"{au}_c" for au in AUS]  # presence (classification) columns

# MediaPipe FaceMesh landmark index pairs used to measure each facial action,
# following the reference paper's Table 2.
FEATURE_PAIRS = {
    "Left Eye Open":         [(398, 382), (384, 381), (385, 380), (386, 374), (387, 373), (388, 390), (466, 249)],
    "Right Eye Open":        [(246, 7), (161, 163), (160, 144), (159, 145), (158, 153), (157, 154), (173, 155)],
    "Jaw Open":              [(80, 170), (81, 140), (82, 171), (13, 175), (312, 396), (311, 369), (310, 395)],
    "Mouth Open":            [(308, 78), (81, 178), (82, 87), (13, 14), (312, 317), (311, 402), (310, 318)],
    "Left Eyebrows Raised":  [(384, 336), (385, 296), (386, 334), (387, 293), (388, 300)],
    "Right Eyebrows Raised": [(161, 70), (160, 63), (159, 105), (158, 66), (157, 107)],
    "Mouth Width":           [(78, 308)],
}

# Order in which geometric features are written (matches Extracted_feautures2.csv).
FEATURE_NAMES = [
    "Left Eye Open",
    "Right Eye Open",
    "Jaw Open",
    "Mouth Open",
    "Left Eyebrows Raised",
    "Right Eyebrows Raised",
    "Mouth Width",
]

N_LANDMARKS = 468  # MediaPipe FaceMesh (without iris refinement)


# --------------------------------------------------------------------------- #
# Geometric feature extraction
# --------------------------------------------------------------------------- #

def euclidean(p1, p2) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def extract_geometric_features(landmarks) -> dict:
    """Compute the 7 ICD-normalised geometric features for one frame.

    `landmarks` is a sequence of (x, y, z) tuples, one per MediaPipe landmark.
    Distances are normalised by the Inter-Canthal Distance (ICD, landmarks
    133 <-> 362) so the features are scale invariant.
    """
    mesh = np.array(landmarks)
    mesh_centered = mesh - mesh.mean(axis=0)

    icd = euclidean(mesh_centered[133], mesh_centered[362])
    if icd == 0:
        icd = 1e-6

    features = {}
    for name in FEATURE_NAMES:
        pairs = FEATURE_PAIRS[name]
        dists = [euclidean(mesh_centered[a], mesh_centered[b]) for a, b in pairs]
        features[name] = float(np.mean(dists)) / icd
    return features


def landmark_columns() -> list:
    """Column names for the raw per-frame landmark CSV (x_i, y_i, z_i)."""
    cols = []
    for i in range(N_LANDMARKS):
        cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    return cols


# --------------------------------------------------------------------------- #
# OpenFace
# --------------------------------------------------------------------------- #

def run_openface(video_path: str, output_dir: str, openface_bin: str) -> str:
    """Run OpenFace FeatureExtraction on one video and return the CSV path.

    Skips the run if the output CSV already exists.
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    openface_csv = os.path.join(output_dir, base + ".csv")

    if os.path.exists(openface_csv):
        return openface_csv

    if not openface_bin or not os.path.exists(openface_bin):
        raise FileNotFoundError(
            f"OpenFace binary not found at '{openface_bin}'. Build OpenFace and "
            "pass its FeatureExtraction path via --openface-bin. "
            "See https://github.com/TadasBaltrusaitis/OpenFace/wiki"
        )

    subprocess.run(
        [openface_bin, "-f", video_path, "-out_dir", output_dir, "-aus"],
        check=True,
    )
    return openface_csv


# --------------------------------------------------------------------------- #
# MediaPipe landmarks
# --------------------------------------------------------------------------- #

def extract_landmarks(video_path: str):
    """Run MediaPipe FaceMesh over a video, returning a list of per-frame rows.

    Each row holds N_LANDMARKS * 3 values (x, y, z interleaved). Frames with no
    detected face are filled with NaN.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "mediapipe and opencv-python are required for landmark extraction. "
            "Install them with: pip install mediapipe opencv-python"
        ) from exc

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            row = []
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z])
            rows.append(row)
        else:
            rows.append([np.nan] * N_LANDMARKS * 3)

    cap.release()
    face_mesh.close()
    return rows


# --------------------------------------------------------------------------- #
# Per-clip statistics
# --------------------------------------------------------------------------- #

def compute_clip_stats(combined: pd.DataFrame, base: str) -> pd.DataFrame:
    """Aggregate a clip's per-frame features into one row of mean/var/entropy."""
    stat_cols = FEATURE_NAMES + AU_R
    values, names = [], []
    for col in stat_cols:
        vals = combined[col].dropna().values
        if len(vals) == 0:
            mean_v, var_v, ent_v = np.nan, np.nan, np.nan
        else:
            hist = np.histogram(vals, bins=10, density=True)[0] + 1e-8
            mean_v, var_v, ent_v = np.mean(vals), np.var(vals), entropy(hist)
        values.extend([mean_v, var_v, ent_v])
        names.extend([f"{col}_mean", f"{col}_var", f"{col}_entropy"])

    row = pd.DataFrame([values], columns=names)
    row.insert(0, "clip", base)
    return row


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def process_video(video_path: str, output_dir: str, openface_bin: str):
    """Process a single video, returning (per_frame_df, clip_stats_df)."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing {base} ...")

    # 1. OpenFace AUs
    openface_csv = run_openface(video_path, output_dir, openface_bin)
    openface_df = pd.read_csv(openface_csv)
    openface_df.columns = [c.strip() for c in openface_df.columns]  # OpenFace pads with spaces
    openface_aus = openface_df[AU_C + AU_R].reset_index(drop=True)

    # 2. MediaPipe landmarks (saved raw for optional visualisation / reuse)
    landmark_rows = extract_landmarks(video_path)
    landmarks_df = pd.DataFrame(landmark_rows, columns=landmark_columns())
    landmarks_df["clip"] = base
    landmarks_df.to_csv(os.path.join(output_dir, f"{base}_landmarks.csv"), index=False)

    # 3. Geometric features per frame
    geo_rows = []
    for row in landmark_rows:
        landmarks = [(row[i * 3], row[i * 3 + 1], row[i * 3 + 2]) for i in range(N_LANDMARKS)]
        if np.isnan(landmarks[0][0]):
            geo_rows.append({k: np.nan for k in FEATURE_NAMES})
        else:
            geo_rows.append(extract_geometric_features(landmarks))
    geo_df = pd.DataFrame(geo_rows, columns=FEATURE_NAMES)

    # 4. Combine (align on the shorter of the two frame counts)
    min_len = min(len(geo_df), len(openface_aus))
    combined = pd.concat(
        [geo_df.iloc[:min_len].reset_index(drop=True),
         openface_aus.iloc[:min_len].reset_index(drop=True)],
        axis=1,
    )
    combined.insert(0, "clip", base)

    # 5. Clip-level statistics
    stats = compute_clip_stats(combined, base)
    return combined, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract smile features from video clips for PD detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video-dir", required=True,
                        help="Directory containing input .mp4 clips.")
    parser.add_argument("--output-dir", default="features_output",
                        help="Directory for OpenFace/MediaPipe outputs and CSVs.")
    parser.add_argument("--openface-bin", default=os.environ.get("OPENFACE_BIN", ""),
                        help="Path to the OpenFace FeatureExtraction binary "
                             "(or set the OPENFACE_BIN environment variable).")
    parser.add_argument("--pattern", default="*.mp4",
                        help="Glob pattern for selecting video files.")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    video_files = sorted(glob.glob(os.path.join(args.video_dir, args.pattern)))
    if not video_files:
        print(f"No videos matching '{args.pattern}' found in {args.video_dir}",
              file=sys.stderr)
        return 1

    all_combined, all_stats = [], []
    for video_path in video_files:
        try:
            combined, stats = process_video(video_path, args.output_dir, args.openface_bin)
            all_combined.append(combined)
            all_stats.append(stats)
        except Exception as exc:  # keep going on a bad clip
            print(f"  !! Skipped {os.path.basename(video_path)}: {exc}", file=sys.stderr)

    if not all_stats:
        print("No clips were processed successfully.", file=sys.stderr)
        return 1

    per_frame_path = os.path.join(args.output_dir, "per_frame_features.csv")
    stats_path = os.path.join(args.output_dir, "clip_stats.csv")
    pd.concat(all_combined, ignore_index=True).to_csv(per_frame_path, index=False)
    pd.concat(all_stats, ignore_index=True).to_csv(stats_path, index=False)

    print(f"\nSaved per-frame features -> {per_frame_path}")
    print(f"Saved clip-level stats   -> {stats_path}")
    print("Add a 'PD' label column (0/1) to the clip stats before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
