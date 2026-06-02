"""Assemble the 3-minute MIRA demo video from generated slides and narration."""

from __future__ import annotations

import json
import shutil
import subprocess
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
NARRATION = ROOT / "narration.txt"


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def make_slide(slide: dict, html_dir: Path, image_dir: Path, segment_dir: Path, width: int, height: int, fps: int) -> Path:
    html_path = html_dir / f"{slide['id']}.html"
    image_path = render_slide_html(slide, html_path, image_dir, width, height)
    output = segment_dir / f"{slide['id']}.mp4"
    duration = str(slide["duration"])
    run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-t",
            duration,
            "-r",
            str(fps),
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0xf4f8fc",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(output),
        ]
    )
    return output


def render_slide_html(slide: dict, html_path: Path, image_dir: Path, width: int, height: int) -> Path:
    body_items = "\n".join(f"<li>{escape(item)}</li>" for item in slide["body"])
    html_path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{
    margin: 0;
    width: {width}px;
    height: {height}px;
    overflow: hidden;
    background: #f4f8fc;
    color: #172033;
    font-family: Inter, Arial, Helvetica, sans-serif;
  }}
  .slide {{
    position: relative;
    width: {width}px;
    height: {height}px;
    padding: 105px 118px;
    box-sizing: border-box;
    background:
      linear-gradient(0deg, rgba(255,255,255,0.62), rgba(255,255,255,0.62)),
      repeating-linear-gradient(90deg, rgba(31,111,191,0.075) 0, rgba(31,111,191,0.075) 2px, transparent 2px, transparent 132px),
      #eef3f8;
  }}
  .rail {{
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 22px;
    background: #1f6fbf;
  }}
  .mark {{
    color: #1f6fbf;
    font-family: Menlo, Consolas, monospace;
    font-size: 32px;
    letter-spacing: 0;
    white-space: pre;
    line-height: 1.0;
    opacity: 0.92;
  }}
  h1 {{
    margin: 34px 0 14px;
    color: #1f6fbf;
    font-size: 104px;
    line-height: 1.0;
    letter-spacing: 0;
  }}
  h2 {{
    margin: 0 0 62px;
    max-width: 1380px;
    color: #172033;
    font-size: 40px;
    font-weight: 500;
    letter-spacing: 0;
  }}
  ul {{
    margin: 0;
    padding-left: 42px;
    max-width: 1220px;
    color: #31445c;
    font-size: 38px;
    line-height: 1.35;
  }}
  li {{
    margin: 17px 0;
  }}
  .footer {{
    position: absolute;
    left: 124px;
    bottom: 70px;
    color: #607086;
    font-size: 24px;
    font-weight: 700;
  }}
  .accent {{
    position: absolute;
    right: 115px;
    bottom: 80px;
    width: 360px;
    height: 180px;
    border: 2px dashed rgba(31, 111, 191, 0.38);
    border-radius: 10px;
    transform: skewY(-8deg);
  }}
</style>
</head>
<body>
  <main class="slide">
    <div class="rail"></div>
    <div class="mark">.--..--.   MIRA</div>
    <h1>{escape(slide["title"])}</h1>
    <h2>{escape(slide["subtitle"])}</h2>
    <ul>{body_items}</ul>
    <div class="accent"></div>
    <div class="footer">MIRA | CS 153 Demo</div>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    run(["qlmanage", "-t", "-s", str(width), "-o", str(image_dir), str(html_path)])
    quicklook_output = image_dir / f"{html_path.name}.png"
    image_path = image_dir / f"{slide['id']}.png"
    if image_path.exists():
        image_path.unlink()
    quicklook_output.rename(image_path)
    run(
        [
            "sips",
            "-z",
            str(height),
            str(width),
            str(image_path),
            "--out",
            str(image_path),
        ]
    )
    return image_path


def make_narration(audio_dir: Path) -> tuple[Path, Path]:
    aiff = audio_dir / "narration.aiff"
    m4a = audio_dir / "narration.m4a"
    run(["say", "-r", "185", "-o", str(aiff), "-f", str(NARRATION)])
    run(["ffmpeg", "-y", "-i", str(aiff), "-c:a", "aac", "-b:a", "160k", str(m4a)])
    return aiff, m4a


def make_subtitles(manifest: dict, output_dir: Path) -> Path:
    srt = output_dir / "subtitles.srt"
    cursor = 0
    blocks = []
    for index, slide in enumerate(manifest["slides"], start=1):
        start = cursor
        end = cursor + int(slide["duration"])
        cursor = end
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_time(start)} --> {format_srt_time(end)}",
                    f"{slide['title']}: {slide['subtitle']}",
                    "",
                ]
            )
        )
    srt.write_text("\n".join(blocks), encoding="utf-8")
    return srt


def format_srt_time(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02},000"


def concat_segments(segments: list[Path], concat_file: Path, output: Path) -> None:
    base_dir = concat_file.parent
    concat_file.write_text(
        "".join(f"file '{segment.relative_to(base_dir)}'\n" for segment in segments),
        encoding="utf-8",
    )
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output)])


def mux_video_audio(video: Path, audio: Path, output: Path, duration: int) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-filter_complex",
            f"[1:a]apad,atrim=0:{duration}[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-t",
            str(duration),
            str(output),
        ]
    )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    width, height = manifest["resolution"]
    fps = int(manifest["fps"])
    duration = int(manifest["target_duration_seconds"])

    generated = ROOT / "generated"
    segment_dir = generated / "segments"
    html_dir = generated / "html"
    image_dir = generated / "images"
    audio_dir = ROOT / "audio"
    output_dir = ROOT / "output"
    clips_dir = ROOT / "clips"
    for directory in (segment_dir, html_dir, image_dir, audio_dir, output_dir, clips_dir):
        directory.mkdir(parents=True, exist_ok=True)

    segments = [make_slide(slide, html_dir, image_dir, segment_dir, width, height, fps) for slide in manifest["slides"]]
    _, narration_m4a = make_narration(audio_dir)
    make_subtitles(manifest, output_dir)

    silent_video = generated / "mira-demo-silent.mp4"
    concat_segments(segments, generated / "concat.txt", silent_video)
    output = ROOT / manifest["output"]
    mux_video_audio(silent_video, narration_m4a, output, duration)

    print(f"\nWrote {output}")
    print("Drop real screen recordings into demo-assets/clips/ and update manifest.json for a clip-based edit.")


if __name__ == "__main__":
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required")
    if not shutil.which("say"):
        raise SystemExit("macOS say is required for narration generation")
    if not shutil.which("qlmanage"):
        raise SystemExit("macOS qlmanage is required for HTML slide rendering")
    main()
