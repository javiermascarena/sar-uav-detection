"""Interactive horizon annotation for focal-length (f_px) calibration.

Shows every frame whose gimbal pitch reading is fresh (it differs from the
previous frame's, so the metadata has just been updated) and lies between
--min-pitch and the drone's horizon-visibility limit, and lets you click points
along the horizon. Frames are shown in an order that interleaves 1-degree pitch
bins, so wherever you stop the annotated set is spread over pitch. Results are
appended to a CSV, so you can stop and resume at any time.

Usage (from the repo root, in a terminal, NOT inside a notebook):
    python scripts/annotate_horizon.py                 # all matching frames
    python scripts/annotate_horizon.py --per-drone 150 # at most 150 per drone

Controls in the image window:
    left click   add a point on the horizon (click 5-10 spread across the width)
    right click  remove last point
    enter        save frame and go to next (needs >= 3 points)
    s            skip frame (horizon not visible / ambiguous)
    q            quit (progress is kept)
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SeaDronesSee_MOT"
ANN = DATA / "annotations"
IMG = DATA / "SeaDronesSee_MOT_jpg_compressed" / "Compressed"
OUT = ROOT / "results" / "horizon_annotations.csv"
# Max gimbal pitch per drone. The horizon is in frame only if pitch < VFOV/2
# (+ a little thanks to barrel distortion): XT2 ~17.0 deg, L1D-20c ~20.1 deg.
# The M100 (X5) never flies below 30.7 deg, so its horizon is never visible and
# it is left out.
MAX_PITCH = {"m210": 20.0, "mavic": 22.0}
FIELDS = ["split", "image_id", "drone", "video", "frame_no", "pitch", "altitude", "points", "status"]


def load_frames(min_pitch: float) -> pd.DataFrame:
    """Frames with a fresh pitch reading and min_pitch <= pitch < MAX_PITCH[drone].

    The pitch metadata is quantised to 0.1 deg and refreshed about every 3-4
    video frames, then held. A frame is "fresh" when the previous frame of the
    same video is in the dataset and has a different pitch, i.e. it is the first
    frame carrying a new reading. Frames whose predecessor is missing are
    dropped because freshness cannot be established.
    """
    rows = []
    for split in ["train", "val", "test"]:
        with (ANN / f"instances_{split}_objects_in_water.json").open(encoding="utf-8") as f:
            ann = json.load(f)
        for im in ann["images"]:
            rows.append({
                "split": split,
                "image_id": im["id"],
                "file": IMG / split / Path(im["file_name"]).with_suffix(".jpg").name,
                "drone": im["source"]["drone"],
                "video": im["source"]["video"],
                "frame_no": im["source"]["frame_no"],
                "pitch": im["meta"]["gimbal_pitch"],
                "altitude": im["meta"]["altitude"],
            })
    df = pd.DataFrame(rows)
    # Video names differ only in case across splits (DJI_0001.mov / .MOV): normalise
    df["video"] = df["video"].str.lower()
    # Freshness is judged on the whole video (all splits, all pitches) before filtering
    df = df.sort_values(["drone", "video", "frame_no"])
    prev = df.groupby(["drone", "video"])[["frame_no", "pitch"]].shift(1)
    fresh = (df["frame_no"] - prev["frame_no"] == 1) & (df["pitch"] != prev["pitch"])
    in_range = (df["pitch"] >= min_pitch) & (df["pitch"] < df["drone"].map(MAX_PITCH))
    return df[fresh & in_range]


def order(df: pd.DataFrame, per_drone: int | None, seed: int) -> pd.DataFrame:
    """Per drone, interleave 1-degree pitch bins (random order inside a bin), so
    any prefix of the list, and hence any --per-drone cut, is spread over pitch."""
    df = df.sample(frac=1, random_state=seed)
    turn = df.groupby(["drone", np.floor(df["pitch"])]).cumcount()
    df = df.assign(turn=turn).sort_values(["drone", "turn", "pitch"], kind="stable").drop(columns="turn")
    return df.groupby("drone").head(per_drone) if per_drone else df


def done_ids() -> set:
    if not OUT.exists():
        return set()
    with OUT.open(encoding="utf-8") as f:
        return {(r["split"], int(r["image_id"])) for r in csv.DictReader(f)}


def append(row: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def annotate(frame: pd.Series) -> tuple[str, list]:
    """Returns (status, points). status in {'ok', 'skip', 'quit'}."""
    pts: list[tuple[float, float]] = []
    state = {"status": None}

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(Image.open(frame["file"]))
    ax.axhline(1080, color="cyan", lw=0.5, ls="--")
    ax.axvline(1920, color="cyan", lw=0.5, ls="--")
    ax.set_title(
        f"{frame['drone']} | {frame['video']} #{frame['frame_no']} | pitch {frame['pitch']:.1f}° "
        f"| click horizon, ENTER=save, s=skip, q=quit, right-click=undo"
    )
    (line,) = ax.plot([], [], "r+-", ms=12, lw=0.8)

    def redraw():
        line.set_data([p[0] for p in pts], [p[1] for p in pts])
        fig.canvas.draw_idle()

    def on_click(e):
        if e.inaxes != ax or fig.canvas.toolbar.mode:  # ignore while zoom/pan active
            return
        if e.button == 1:
            pts.append((round(e.xdata, 1), round(e.ydata, 1)))
        elif e.button == 3 and pts:
            pts.pop()
        pts.sort()
        redraw()

    def on_key(e):
        if e.key == "enter" and len(pts) >= 3:
            state["status"] = "ok"
        elif e.key == "s":
            state["status"] = "skip"
        elif e.key == "q":
            state["status"] = "quit"
        if state["status"]:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()
    return state["status"] or "quit", pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-drone", type=int, default=None, help="max frames per drone (default: all)")
    p.add_argument("--min-pitch", type=float, default=5.0, help="lowest gimbal pitch to use, in degrees")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    frames = order(load_frames(args.min_pitch), args.per_drone, args.seed)
    done = done_ids()
    is_done = pd.Series([(s, i) in done for s, i in zip(frames["split"], frames["image_id"])], index=frames.index)
    todo = frames[~is_done]
    print(frames.groupby(["drone", "video"])["pitch"].agg(["size", "min", "max"]).round(1))
    print(pd.DataFrame({"selected": frames.groupby("drone").size(), "remaining": todo.groupby("drone").size()})
          .fillna(0).astype(int))

    for _, fr in todo.iterrows():
        status, pts = annotate(fr)
        if status == "quit":
            break
        append({k: fr[k] for k in FIELDS[:7]} | {"points": json.dumps(pts), "status": status})
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
