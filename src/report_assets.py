from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt


def resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def create_risk_score_timeline(project_root: Path, frame_log_path: str, output_path: str) -> Path:
    frame_log_path = resolve_path(project_root, frame_log_path)
    output_path = resolve_path(project_root, output_path)

    if not frame_log_path.exists():
        raise FileNotFoundError(f"Frame log not found: {frame_log_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(frame_log_path)

    plt.figure(figsize=(12, 5))
    plt.plot(df["time_sec"], df["risk_score"], linewidth=2)
    plt.axhline(25, linestyle="--", linewidth=1, label="MEDIUM threshold")
    plt.axhline(50, linestyle="--", linewidth=1, label="HIGH threshold")
    plt.axhline(75, linestyle="--", linewidth=1, label="DANGER threshold")

    danger_df = df[df["risk_level"] == "DANGER"]
    high_df = df[df["risk_level"] == "HIGH"]

    if len(high_df) > 0:
        plt.scatter(high_df["time_sec"], high_df["risk_score"], label="HIGH samples", s=35)

    if len(danger_df) > 0:
        plt.scatter(danger_df["time_sec"], danger_df["risk_score"], label="DANGER samples", s=50)

    plt.title("Image-space Surrogate Risk Score Timeline")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Risk score (0–100)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def extract_representative_screenshots(project_root: Path, final_video_path: str, event_log_path: str, output_dir: str) -> list[Path]:
    final_video_path = resolve_path(project_root, final_video_path)
    event_log_path = resolve_path(project_root, event_log_path)
    output_dir = resolve_path(project_root, output_dir)

    if not final_video_path.exists():
        raise FileNotFoundError(f"Final video not found: {final_video_path}")

    if not event_log_path.exists():
        raise FileNotFoundError(f"Event log not found: {event_log_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    log_df = pd.read_csv(event_log_path)

    selected = []

    danger_df = log_df[log_df["risk_level"] == "DANGER"]
    high_df = log_df[log_df["risk_level"] == "HIGH"]

    danger_count = 1
    for _, row in danger_df.head(3).iterrows():
        selected.append((f"danger_example_{danger_count}.jpg", int(row["frame_id"])))
        danger_count += 1

    high_count = 1
    for _, row in high_df.head(2).iterrows():
        selected.append((f"high_example_{high_count}.jpg", int(row["frame_id"])))
        high_count += 1

    cap = cv2.VideoCapture(str(final_video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open final video: {final_video_path}")

    saved_paths = []

    for filename, frame_id in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame {frame_id}")
            continue

        output_path = output_dir / filename
        cv2.imwrite(str(output_path), frame)
        saved_paths.append(output_path)
        print(f"Saved frame {frame_id} -> {output_path}")

    cap.release()

    return saved_paths
