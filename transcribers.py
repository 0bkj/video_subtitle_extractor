from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from sysconfig import get_path
from typing import Any, Callable, Protocol


ProgressCallback = Callable[[float], None]
DLL_DIRECTORY_HANDLES = []


def configure_cuda_libraries() -> None:
    if sys.platform != "win32":
        return
    package_root = Path(get_path("purelib")) / "nvidia"
    for component in ("cublas", "cudnn"):
        library_dir = package_root / component / "bin"
        if library_dir.is_dir():
            DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(library_dir)))
            os.environ["PATH"] = f"{library_dir}{os.pathsep}{os.environ.get('PATH', '')}"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


class Transcriber(Protocol):
    def transcribe(
        self, audio_path: Path, progress_callback: ProgressCallback | None = None
    ) -> list[Segment]: ...


class FasterWhisperTranscriber:
    def __init__(self, config: dict):
        if config["device"] in {"cuda", "auto"}:
            configure_cuda_libraries()
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "未安装 faster-whisper。请运行 run.bat / run.sh，或先安装 requirements.txt。"
            ) from error

        self.config = config
        self.model_class = WhisperModel
        self.cpu_fallback = False
        try:
            self.model = self._load(config["device"], config["compute_type"])
        except (OSError, RuntimeError) as error:
            if not self._can_fallback(error):
                raise
            self._switch_to_cpu()
        print(f"  模型运行设备: {self.model.model.device}")

    def _load(self, device: str, compute_type: str):
        return self.model_class(self.config["model"], device=device, compute_type=compute_type)

    def _can_fallback(self, error: Exception) -> bool:
        message = str(error).lower()
        return self.config["device"] == "auto" and not self.cpu_fallback and any(
            token in message for token in ("cuda", "cublas", "cudnn")
        )

    def _switch_to_cpu(self) -> None:
        print("  自动设备的 CUDA 环境不可用，改用 CPU (int8) 重试")
        self.cpu_fallback = True
        self.model = self._load("cpu", "int8")

    def transcribe(
        self, audio_path: Path, progress_callback: ProgressCallback | None = None
    ) -> list[Segment]:
        try:
            segments, info = self._decode(audio_path, progress_callback)
        except (OSError, RuntimeError) as error:
            if not self._can_fallback(error):
                raise
            if progress_callback and sys.stdout.isatty():
                print()
            self._switch_to_cpu()
            if progress_callback:
                progress_callback(0.0)
            segments, info = self._decode(audio_path, progress_callback)
        if progress_callback:
            progress_callback(1.0)
        print(f"  识别语言: {info.language} (概率 {info.language_probability:.2f})")
        return segments

    def _decode(
        self, audio_path: Path, progress_callback: ProgressCallback | None
    ) -> tuple[list[Segment], Any]:
        raw_segments, info = self.model.transcribe(
            str(audio_path),
            language=self.config["language"],
            beam_size=self.config["beam_size"],
            vad_filter=self.config["vad_filter"],
            task="transcribe",
        )
        segments = []
        for segment in raw_segments:
            if segment.text.strip():
                segments.append(Segment(segment.start, segment.end, segment.text.strip()))
            if progress_callback and info.duration > 0:
                progress_callback(segment.end / info.duration)
        return segments, info


def create_transcriber(config: dict) -> Transcriber:
    backend = config["backend"]
    if backend == "faster-whisper":
        return FasterWhisperTranscriber(config)
    if backend == "qwen-asr":
        raise RuntimeError(
            "qwen-asr 后端尚未实现。可在 transcribers.py 中添加实现并接入 create_transcriber。"
        )
    raise ValueError(f"不支持的转写后端: {backend}")
