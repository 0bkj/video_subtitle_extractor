from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, deque
from pathlib import Path

from transcribers import Segment, create_transcriber


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".model_cache"))
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".m4v", ".ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".amr"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".ass", ".ssa")
TEXT_SUBTITLE_CODECS = {"subrip", "mov_text", "ass", "ssa", "webvtt", "text"}
TIME_RANGE = re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")
TAG = re.compile(r"<[^>]+>")


class ProgressBar:
    def __init__(self, label: str):
        self.label = label
        self.last_percent = -1
        self.interactive = sys.stdout.isatty()

    def update(self, fraction: float) -> None:
        percent = max(0, min(100, int(fraction * 100)))
        if percent == self.last_percent:
            return
        previous = self.last_percent
        self.last_percent = percent
        if not self.interactive and percent not in (0, 100) and percent // 10 == previous // 10:
            return
        filled = percent * 24 // 100
        bar = "#" * filled + "-" * (24 - filled)
        ending = "\n" if percent == 100 or not self.interactive else ""
        prefix = "\r" if self.interactive else ""
        print(f"{prefix}  {self.label} [{bar}] {percent:3d}%", end=ending, flush=True)

    def close(self) -> None:
        if self.interactive and self.last_percent != 100:
            print(flush=True)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    defaults = {
        "backend": "faster-whisper",
        "model": "small",
        "device": "auto",
        "compute_type": "auto",
        "language": None,
        "beam_size": 5,
        "vad_filter": True,
    }
    defaults.update(config)
    if defaults["backend"] not in {"faster-whisper", "qwen-asr"}:
        raise ValueError("backend 应为 faster-whisper 或 qwen-asr")
    if not isinstance(defaults["model"], str) or not defaults["model"].strip():
        raise ValueError("model 必须是非空字符串")
    if defaults["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 应为 auto、cpu 或 cuda")
    if defaults["language"] is not None and not isinstance(defaults["language"], str):
        raise ValueError("language 应为 null 或语言代码，例如 zh、en")
    if isinstance(defaults["beam_size"], bool) or not isinstance(defaults["beam_size"], int) or defaults["beam_size"] < 1:
        raise ValueError("beam_size 必须是正整数")
    if not isinstance(defaults["vad_filter"], bool):
        raise ValueError("vad_filter 必须是 true 或 false")
    return defaults


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"命令执行失败: {detail[-1200:]}")
    return result.stdout


def find_sidecar(media: Path) -> Path | None:
    candidates = [
        entry for entry in media.parent.iterdir()
        if entry.is_file() and entry.stem.casefold() == media.stem.casefold()
        and entry.suffix.lower() in SUBTITLE_EXTENSIONS
    ]
    return min(candidates, key=lambda entry: SUBTITLE_EXTENSIONS.index(entry.suffix.lower()), default=None)


def subtitle_streams(media: Path) -> list[int]:
    probe = run_command([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(media)
    ])
    streams = json.loads(probe).get("streams", [])
    return [
        stream["index"] for stream in streams
        if stream.get("codec_type") == "subtitle"
        and stream.get("codec_name") in TEXT_SUBTITLE_CODECS
    ]


def convert_subtitle(source: Path, destination: Path, stream_index: int | None = None) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    command += ["-map", f"0:{stream_index}" if stream_index is not None else "0:s:0"]
    command += ["-c:s", "srt", str(destination)]
    run_command(command)


def subtitle_text(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = []
    for block in re.split(r"\r?\n\s*\r?\n", content):
        parts = block.strip().splitlines()
        timing = next((index for index, part in enumerate(parts) if TIME_RANGE.search(part)), None)
        if timing is None:
            continue
        caption = " ".join(parts[timing + 1 :])
        caption = re.sub(r"\s+", " ", html.unescape(TAG.sub("", caption))).strip()
        if caption:
            lines.append(caption)
    return lines


def extract_existing_subtitle(media: Path, destination: Path) -> list[str]:
    sidecar = find_sidecar(media)
    if sidecar:
        try:
            convert_subtitle(sidecar, destination)
            lines = subtitle_text(destination)
            if lines:
                print(f"  使用同名字幕: {sidecar.name}")
                return lines
        except (RuntimeError, OSError) as error:
            print(f"  同名字幕不可用: {error}")

    for stream_index in subtitle_streams(media):
        try:
            convert_subtitle(media, destination, stream_index)
            lines = subtitle_text(destination)
            if lines:
                print(f"  使用内嵌文字字幕流: {stream_index}")
                return lines
        except (RuntimeError, OSError) as error:
            print(f"  字幕流 {stream_index} 不可用: {error}")
    return []


def media_duration(media: Path) -> float | None:
    try:
        output = run_command([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(media),
        ])
        duration = float(output.strip())
        return duration if duration > 0 else None
    except (RuntimeError, ValueError):
        return None


def extract_audio(media: Path, destination: Path, progress: ProgressBar) -> None:
    duration = media_duration(media)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", "-progress", "pipe:1", "-nostats", str(destination),
    ]
    recent_output = deque(maxlen=30)
    progress.update(0.0)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    try:
        for line in process.stdout:
            recent_output.append(line.strip())
            if duration and line.startswith("out_time="):
                try:
                    hours, minutes, seconds = line.partition("=")[2].strip().split(":")
                    elapsed = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                    progress.update(elapsed / duration)
                except ValueError:
                    pass
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg 音频提取失败: {' '.join(recent_output)[-1200:]}")
        progress.update(1.0)
    finally:
        progress.close()


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def write_transcription(segments: list[Segment], destination: Path) -> list[str]:
    lines = []
    captions = []
    for segment in segments:
        text = re.sub(r"\s+", " ", segment.text).strip()
        if not text:
            continue
        lines.append(text)
        captions.append(
            f"{len(captions) + 1}\n{srt_timestamp(segment.start)} --> "
            f"{srt_timestamp(max(segment.end, segment.start + 0.001))}\n{text}\n"
        )
    destination.write_text("\n".join(captions), encoding="utf-8-sig")
    return lines


def output_stem(media: Path, counts: Counter) -> str:
    return media.stem if counts[media.stem.casefold()] == 1 else f"{media.stem}__{media.suffix[1:].lower()}"


def main() -> int:
    parser = argparse.ArgumentParser(description="批量提取音视频字幕，或使用 Whisper 转写语音")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("未找到 ffmpeg / ffprobe。请安装 ffmpeg 并加入 PATH。")
        input_dir = ROOT / "input_videos"
        output_dir = ROOT / "output_text"
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        media_files = sorted(
            (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS),
            key=lambda path: path.name.casefold(),
        )
        if not media_files:
            print(f"没有找到音视频文件。请将文件放入 {input_dir}")
            return 0
        counts = Counter(media.stem.casefold() for media in media_files)
        transcriber = None
        failures = 0
        for index, media in enumerate(media_files, start=1):
            print(f"[{index}/{len(media_files)}] 处理: {media.name}", flush=True)
            try:
                with tempfile.TemporaryDirectory(prefix="subtitle_", dir=output_dir) as temp_name:
                    temp_dir = Path(temp_name)
                    temp_srt = temp_dir / "result.srt"
                    lines = extract_existing_subtitle(media, temp_srt)
                    if not lines:
                        print("  未找到可用文字字幕，开始语音转写", flush=True)
                        audio = temp_dir / "audio.wav"
                        extract_audio(media, audio, ProgressBar("提取音频"))
                        if transcriber is None:
                            print("  加载语音模型...", flush=True)
                            transcriber = create_transcriber(config)
                        progress = ProgressBar("语音识别")
                        progress.update(0.0)
                        try:
                            segments = transcriber.transcribe(audio, progress.update)
                            progress.update(1.0)
                        finally:
                            progress.close()
                        lines = write_transcription(segments, temp_srt)
                        if not lines:
                            raise RuntimeError("转写没有得到文字")
                    temp_txt = temp_dir / "result.txt"
                    temp_txt.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
                    stem = output_stem(media, counts)
                    os.replace(temp_srt, output_dir / f"{stem}.srt")
                    os.replace(temp_txt, output_dir / f"{stem}.txt")
                    print(f"  完成: {stem}.txt / {stem}.srt", flush=True)
            except Exception as error:
                failures += 1
                print(f"  失败: {error}", file=sys.stderr, flush=True)
        print(f"完成 {len(media_files) - failures}/{len(media_files)} 个文件。结果目录: {output_dir}")
        return 1 if failures else 0
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
