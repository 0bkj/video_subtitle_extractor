# 批量音视频字幕提取

把视频或录音放入 `input_videos`，运行脚本，`output_text` 中会生成同名 `.txt` 和带时间轴的 `.srt`。文件夹名称沿用 `input_videos`，音频直接放在里面即可。

## 准备

- Python 3.10 或更新版本。
- `ffmpeg` 和 `ffprobe`，两者均需在系统 `PATH` 中。运行 `ffmpeg -version` 可检查。
- 首次语音转写若本地没有所选 Whisper 模型，需要联网下载。模型默认缓存到项目的 `.model_cache`；已有文字字幕无需下载模型。
- Windows 使用 NVIDIA GPU 时，需要 CUDA 12 的 cuBLAS 和 cuDNN 9。`run.bat` 在缺少这些依赖时会从 PyPI 安装到项目虚拟环境，不修改系统 CUDA 安装。

## 使用

1. 将视频（`.mp4`、`.mkv`、`.mov`、`.webm`、`.avi`、`.flv`、`.m4v`、`.ts`）或音频（`.mp3`、`.wav`、`.m4a`、`.flac`、`.ogg`、`.opus`、`.aac`、`.wma`、`.amr`）放入 `input_videos`。目前只扫描此文件夹的第一层。
2. Windows 双击 `run.bat`；macOS/Linux 在终端运行 `sh run.sh`。启动脚本会创建 `.venv`，并在需要时安装依赖。
3. 到 `output_text` 查看结果。同名文件每次运行会覆盖先前结果。若多个文件仅扩展名不同，输出名会附上扩展名，例如 `讲座__mp4.txt`。

运行时会显示当前文件序号，以及音频提取、语音识别的进度条。语音识别进度根据模型已处理的音频时间估算，长片段之间可能暂时不变；首次下载模型时会先显示“加载语音模型”。

也可以手动安装后运行：

```sh
python -m venv .venv
# Windows: .venv\Scripts\python -m pip install -r requirements.txt
# Windows: .venv\Scripts\python run.py
# macOS/Linux:
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

## 字幕优先级

先尝试与音视频同名的 `.srt`、`.vtt`、`.ass` 或 `.ssa` 文件，再尝试文件中的文字字幕流。字幕会统一转换为 SRT，并提取纯文字到 TXT。图片字幕（如 PGS、VobSub）无法直接提取文字，此时如有音轨，会改用语音转写。所有已有字幕路径均在本地通过 ffmpeg 处理；本项目不从哔哩哔哩网页或 API 下载字幕。

## 配置

编辑 `config.json`：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `backend` | `faster-whisper` | 当前可用的转写后端；`qwen-asr` 是预留接口，尚未实现 |
| `model` | `large-v3` | Whisper 模型名，如 `tiny`、`base`、`small`、`medium`、`large-v3` |
| `device` | `cuda` | `auto`、`cpu` 或 `cuda`；明确指定 `cuda` 时 GPU 出错会报告失败，不会静默改用 CPU |
| `compute_type` | `float16` | CTranslate2 计算类型；8 GB 显存不足时可试 `int8_float16`，CPU 可用 `int8` |
| `language` | `null` | 自动判断语言；可指定 `zh` 或 `en` |
| `beam_size` | `5` | 搜索宽度 |
| `vad_filter` | `true` | 尝试过滤静音 |

中英混合建议保留 `language: null`；自动语言判断通常以整段音频为主，代码切换的准确率取决于音质和模型，必要时改用更大的模型。要扩展 Qwen ASR，在 `transcribers.py` 实现 `Transcriber` 接口并在 `create_transcriber` 注册。当前把 `backend` 设为 `qwen-asr` 会给出明确错误，不会默默改用 Whisper。

Windows GPU 依赖固定在 `requirements-gpu-windows.txt`，可手动运行 `.venv\Scripts\python -m pip install --index-url https://pypi.org/simple -r requirements-gpu-windows.txt`。Linux 使用 CUDA 时需自行安装 CUDA 12 的 cuBLAS 与 cuDNN 9，并使其位于动态库搜索路径。首次下载 `large-v3` 比 `small` 更占磁盘空间与时间。

在没有兼容 NVIDIA GPU 的机器上使用时，把 `device` 改为 `cpu`、`compute_type` 改为 `int8`。切换模型后，首次使用的新模型也需要单独下载。

每个音视频文件单独处理；一个文件失败不会中止其他文件，运行结束会汇总成功数，并以非零退出码报告失败。只有找到可用字幕或成功转写后才写入该文件的输出。首次下载模型较大，所需时间取决于网络。
