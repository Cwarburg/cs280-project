# extract_frames.py
"""
Extract frames at 2 fps from all SoccerNet .mkv files.

Directory structure produced:
  frames/{season}/{game_slug}/{half}/  e.g. frames/2015-2016/Chelsea_2-2_Swansea/1/

Usage:
    python extract_frames.py [--src SoccerNet/england_epl] [--out frames] [--fps 2]
"""
import argparse
import subprocess
from pathlib import Path


def game_slug(game_dir: Path) -> str:
    """Turn '2015-08-08 - 19-30 Chelsea 2 - 2 Swansea' into 'Chelsea_2-2_Swansea'."""
    import re
    name = game_dir.name
    # Strip leading date (YYYY-MM-DD) and time (HH-MM) prefix
    name = re.sub(r'^\d{4}-\d{2}-\d{2} - \d{2}-\d{2} ', '', name)
    # 'Chelsea 2 - 2 Swansea' → 'Chelsea_2-2_Swansea'
    name = re.sub(r' - ', '-', name)
    return name.replace(" ", "_")


def extract(mkv_path: Path, out_dir: Path, fps: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.jpg")):
        print(f"  skip {out_dir} (already extracted)")
        return
    cmd = [
        "ffmpeg", "-i", str(mkv_path),
        "-vf", f"fps={fps}",
        "-q:v", "3",
        str(out_dir / "%06d.jpg"),
        "-hide_banner", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    n = len(list(out_dir.glob("*.jpg")))
    print(f"  {n} frames → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("SoccerNet/england_epl"))
    p.add_argument("--out", type=Path, default=Path("frames"))
    p.add_argument("--fps", type=int, default=2)
    args = p.parse_args()

    mkv_files = sorted(args.src.rglob("*.mkv"))
    print(f"Found {len(mkv_files)} videos")

    for mkv in mkv_files:
        season   = mkv.parts[-3]          # e.g. '2015-2016'
        game_dir = mkv.parent
        slug     = game_slug(game_dir)
        half     = mkv.stem.split("_")[0] # '1' or '2'
        out_dir  = args.out / season / slug / half
        print(f"Processing {season}/{slug}/{half}")
        extract(mkv, out_dir, args.fps)


if __name__ == "__main__":
    main()
