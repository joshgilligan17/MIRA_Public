# MIRA Demo Video Assets

This folder contains a reproducible 3-minute demo-video kit.

Generated outputs:

- `output/mira-demo.mp4`: assembled 3-minute video
- `audio/narration.aiff`: narration generated with macOS `say`
- `audio/narration.m4a`: compressed narration track
- `generated/segments/`: generated title/section video slides
- `output/subtitles.srt`: simple segment captions

Editable inputs:

- `manifest.json`: slide order, timing, and title-card text
- `narration.txt`: narration script
- `make_demo_video.py`: build script

Run:

```bash
cd <repo-root>
uv run python demo-assets/make_demo_video.py
```

To replace a generated slide with a real screen recording later, add the clip to
`demo-assets/clips/` and update `manifest.json`.
