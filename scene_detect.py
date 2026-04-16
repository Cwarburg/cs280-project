"""
Detect camera cuts in extracted frame directories using PySceneDetect and save cut indices to JSON.

Usage:
    python scene_detect.py --frames_root frames --out cuts.json
"""
import argparse
import json
from pathlib import Path

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


def detect_cuts(frame_dir: Path, threshold: float = 27.0, framerate: float = 2.0) -> list[int]:
    """
    Return list of cut indices i such that a camera cut occurs between
    frame i and frame i+1. Uses PySceneDetect ContentDetector on an image sequence.
    """
    frames = sorted(frame_dir.glob("*.jpg"))
    if len(frames) < 2:
        return []

    # Build printf-style pattern from the first filename (e.g. %06d.jpg)
    first = frames[0].name
    stem_len = len(frames[0].stem)
    pattern = str(frame_dir / f"%0{stem_len}d.jpg")

    video = open_video(pattern, framerate=framerate, backend="opencv")
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()

    # A cut occurs at the start frame of every scene after the first
    cuts = [scene[0].get_frames() - 1 for scene in scene_list[1:]]
    return cuts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", default="frames", type=Path)
    parser.add_argument("--out", default="cuts.json", type=Path)
    parser.add_argument("--threshold", default=27.0, type=float,
                        help="ContentDetector threshold (default 27.0)")
    parser.add_argument("--framerate", default=2.0, type=float,
                        help="Framerate to assign to image sequences (default 2.0)")
    args = parser.parse_args()

    result = {}
    for half_dir in sorted(args.frames_root.rglob("*")):
        if not half_dir.is_dir():
            continue
        jpgs = list(half_dir.glob("*.jpg"))
        if not jpgs:
            continue
        print(f"Processing {half_dir} ({len(jpgs)} frames)...")
        cuts = detect_cuts(half_dir, threshold=args.threshold, framerate=args.framerate)
        key = str(half_dir.relative_to(args.frames_root))
        result[key] = cuts
        print(f"  found {len(cuts)} cuts")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
