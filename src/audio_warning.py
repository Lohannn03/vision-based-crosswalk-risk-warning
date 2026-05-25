from pathlib import Path
import subprocess

import cv2
import pandas as pd
from gtts import gTTS
from pydub import AudioSegment


def resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def create_voice_warning_audio(
    project_root: Path,
    output_video_path: str,
    event_log_path: str,
    audio_output_path: str,
    temp_audio_dir: str,
    high_text: str,
    danger_text: str,
    high_cooldown_sec: float = 4.0,
    danger_cooldown_sec: float = 3.0,
) -> Path:
    output_video_path = resolve_path(project_root, output_video_path)
    event_log_path = resolve_path(project_root, event_log_path)
    audio_output_path = resolve_path(project_root, audio_output_path)
    temp_audio_dir = resolve_path(project_root, temp_audio_dir)

    if not output_video_path.exists():
        raise FileNotFoundError(f"Output video not found: {output_video_path}")

    if not event_log_path.exists():
        raise FileNotFoundError(f"Event log not found: {event_log_path}")

    temp_audio_dir.mkdir(parents=True, exist_ok=True)
    audio_output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(output_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps if fps > 0 else 60
    cap.release()

    log_df = pd.read_csv(event_log_path)

    high_mp3 = temp_audio_dir / "high_warning.mp3"
    danger_mp3 = temp_audio_dir / "danger_warning.mp3"

    gTTS(text=high_text, lang="en").save(str(high_mp3))
    gTTS(text=danger_text, lang="en").save(str(danger_mp3))

    high_voice = AudioSegment.from_mp3(str(high_mp3)) + 3
    danger_voice = AudioSegment.from_mp3(str(danger_mp3)) + 5

    base_audio = AudioSegment.silent(duration=int(duration_sec * 1000))

    last_high_audio_time = -999.0
    last_danger_audio_time = -999.0

    for _, row in log_df.iterrows():
        risk = row["risk_level"]
        event_time = float(row["time_sec"])
        position_ms = int(event_time * 1000)

        if risk == "DANGER":
            if event_time - last_danger_audio_time >= danger_cooldown_sec:
                base_audio = base_audio.overlay(danger_voice, position=position_ms)
                last_danger_audio_time = event_time

        elif risk == "HIGH":
            if event_time - last_high_audio_time >= high_cooldown_sec:
                base_audio = base_audio.overlay(high_voice, position=position_ms)
                last_high_audio_time = event_time

    base_audio.export(str(audio_output_path), format="wav")
    return audio_output_path


def merge_audio_to_video_h264(
    project_root: Path,
    input_video_path: str,
    audio_path: str,
    output_video_path: str,
) -> Path:
    input_video_path = resolve_path(project_root, input_video_path)
    audio_path = resolve_path(project_root, audio_path)
    output_video_path = resolve_path(project_root, output_video_path)

    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_video_path),
        "-i", str(audio_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_video_path),
    ]

    subprocess.run(cmd, check=True)
    return output_video_path
