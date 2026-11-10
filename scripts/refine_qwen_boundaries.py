#!/usr/bin/env python3
"""Two-pass, event-specific temporal boundary refinement using a local Qwen-VL server.

This tool never modifies a LeRobot dataset. It writes the Qwen responses and
selected timestamps to JSON, so they can be evaluated against manual labels.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import av
import requests
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq


BOUNDARIES = (
    (
        "pick_to_move",
        "The pickup phase includes reaching, grasping, and lifting the cube. "
        "Find the instant the robot begins carrying the held cube toward the drawer, "
        "not the earlier instant when the gripper merely closes.",
    ),
    (
        "move_to_open",
        "Find the first visible motion that opens the drawer. The preceding phase is "
        "transporting the cube toward the drawer; do not select the arm approaching it.",
    ),
    (
        "open_to_place",
        "Find when the drawer has finished opening and the robot begins the placement "
        "of the cube into the open drawer.",
    ),
    (
        "place_to_close",
        "Find when the cube has been released inside the drawer and the closing motion "
        "of the drawer begins.",
    ),
)


def episode_metadata(root: Path, episode: int) -> dict:
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            if int(row["episode_index"]) == episode:
                return row
    raise ValueError(f"Episode {episode} was not found")


def video_context(root: Path, episode: int, camera_key: str) -> tuple[Path, float, float]:
    info = json.loads((root / "meta" / "info.json").read_text())
    metadata = episode_metadata(root, episode)
    prefix = f"videos/{camera_key}/"
    path = root / info["video_path"].format(
        video_key=camera_key,
        chunk_index=int(metadata[f"{prefix}chunk_index"]),
        file_index=int(metadata[f"{prefix}file_index"]),
    )
    return path, float(metadata[f"{prefix}from_timestamp"]), float(metadata[f"{prefix}to_timestamp"])


def frames_at(video_path: Path, wanted: list[float]) -> list[tuple[float, Image.Image]]:
    """Decode the closest software-decoded AV1 frame for every requested timestamp."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    wanted = sorted(wanted)
    selected: list[tuple[float, Image.Image]] = []
    pending = 0
    previous: tuple[float, Image.Image] | None = None
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        now = float(frame.time)
        while pending < len(wanted) and now >= wanted[pending]:
            candidate = (now, frame.to_image().convert("RGB"))
            if previous is not None and abs(previous[0] - wanted[pending]) <= abs(now - wanted[pending]):
                candidate = previous
            selected.append(candidate)
            pending += 1
        if pending == len(wanted):
            break
        previous = (now, frame.to_image().convert("RGB"))
    container.close()
    if pending != len(wanted):
        raise RuntimeError(f"Could not decode all requested frames from {video_path}")
    return selected


def contact_sheets(frames: list[tuple[float, Image.Image]], columns: int = 4, tile_width: int = 320) -> list[str]:
    """Make timestamp-labelled JPEG sheets and return them as data URLs."""
    font = ImageFont.load_default()
    sheets: list[str] = []
    for offset in range(0, len(frames), columns * 2):
        group = frames[offset : offset + columns * 2]
        tiles: list[Image.Image] = []
        for timestamp, image in group:
            height = round(image.height * tile_width / image.width)
            tile = image.resize((tile_width, height))
            canvas = Image.new("RGB", (tile_width, height + 28), "black")
            canvas.paste(tile, (0, 28))
            ImageDraw.Draw(canvas).text((8, 7), f"{timestamp:05.2f}s", fill="white", font=font)
            tiles.append(canvas)
        height = max(tile.height for tile in tiles)
        sheet = Image.new("RGB", (columns * tile_width, 2 * height), "black")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % columns) * tile_width, (i // columns) * height))
        buffer = io.BytesIO()
        sheet.save(buffer, format="JPEG", quality=90)
        sheets.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode())
    return sheets


def parse_json(content: str) -> dict:
    content = content.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in response: {content[:300]!r}")
    return json.loads(match.group())


def ask_boundary(api_base: str, model_id: str, images: list[str], instruction: str, lo: float, hi: float) -> dict:
    prompt = f"""You are locating one temporal event in a robot video. {instruction}

The images are in chronological order and each frame has a printed timestamp. Select the closest displayed timestamp in the inclusive interval [{lo:.2f}, {hi:.2f}].
Return ONLY valid JSON: {{\"timestamp\": <number>, \"reason\": \"brief visual evidence\"}}. Do not use Markdown."""
    content = [{"type": "image_url", "image_url": {"url": image}} for image in images]
    content.append({"type": "text", "text": prompt})
    response = requests.post(
        api_base.rstrip("/") + "/chat/completions",
        json={"model": model_id, "messages": [{"role": "user", "content": content}], "temperature": 0.0, "max_tokens": 128},
        timeout=300,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    parsed = parse_json(raw)
    timestamp = float(parsed["timestamp"])
    return {"timestamp": min(max(timestamp, lo), hi), "reason": parsed.get("reason", ""), "raw": raw}


def grid(lo: float, hi: float, fps: float) -> list[float]:
    count = max(2, int(round((hi - lo) * fps)) + 1)
    return [lo + i * (hi - lo) / (count - 1) for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--camera-key", default="observation.images.cam_low")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--global-fps", type=float, default=1.0)
    parser.add_argument("--refine-fps", type=float, default=4.0)
    parser.add_argument("--refine-radius-s", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    video_path, episode_start, episode_end = video_context(args.root, args.episode, args.camera_key)
    result = {"episode": args.episode, "camera_key": args.camera_key, "video": str(video_path), "boundaries": {}}
    for name, instruction in BOUNDARIES:
        global_frames = frames_at(video_path, grid(episode_start, episode_end, args.global_fps))
        coarse = ask_boundary(args.api_base, args.model_id, contact_sheets(global_frames), instruction, episode_start, episode_end)
        lo = max(episode_start, coarse["timestamp"] - args.refine_radius_s)
        hi = min(episode_end, coarse["timestamp"] + args.refine_radius_s)
        local_frames = frames_at(video_path, grid(lo, hi, args.refine_fps))
        refined = ask_boundary(args.api_base, args.model_id, contact_sheets(local_frames), instruction, lo, hi)
        result["boundaries"][name] = {"coarse": coarse, "refined": refined}
        print(f"{name}: {coarse['timestamp']:.2f}s -> {refined['timestamp']:.2f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
