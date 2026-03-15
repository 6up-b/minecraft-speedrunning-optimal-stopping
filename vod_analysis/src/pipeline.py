"""
Run-aware VOD analysis pipeline.

Stage 1:
    Scan the VOD at a low sample rate and split it into rough runs by
    detecting near-zero timers (00:0x.xxx). Timer reads use a cached timer ROI
    when available and re-run anchor matching if confidence drops.

Stage 2:
    For each run, search for the first occurrence of each toast in order at a
    higher sample rate. Toast detection is gated by an HSV green-text filter,
    then only the expected template is matched.

Output columns:
    run  toast_index  timestamp  frame  timer_text  timer_conf
    template_name  template_score  timer_source

Usage (from vod_analysis/):
    python -m src.pipeline --video part538.mp4
"""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.timer.infer import infer_timer, infer_timer_from_bbox, load_timer_layout
from src.timer.model import TinyCharCNN
from src.toast.green_crop import green_text_crop, load_hsv_config
from src.toast.template_match import (
    ToastMatchResult,
    ToastTemplate,
    crop_toast_roi,
    load_toast_templates,
    match_single_template,
)

MATCH_SCALES = (0.95, 1.00, 1.05)


@dataclass
class RunWindow:
    run_idx: int
    start_frame: int
    end_frame: int


@dataclass
class PipelineEvent:
    run_idx: int
    toast_index: int
    timestamp: str
    frame: int
    timer_text: str
    timer_conf: float
    template_name: str
    template_score: float
    timer_source: str


def fmt_optional_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


class TimerTracker:
    def __init__(
        self,
        model: TinyCharCNN,
        layout,
        device: str,
        anchor_template: np.ndarray,
        refresh_conf_threshold: float,
        conf_drop_delta: float,
        cache_accept_conf: float,
    ):
        self.model = model
        self.layout = layout
        self.device = device
        self.anchor_template = anchor_template
        self.refresh_conf_threshold = refresh_conf_threshold
        self.conf_drop_delta = conf_drop_delta
        self.cache_accept_conf = cache_accept_conf
        self.cached_timer_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_conf: Optional[float] = None

    def _update_cache(self, result: Dict) -> None:
        conf = float(result.get("conf", 0.0) or 0.0)
        bbox = result.get("timer_bbox")
        if bbox is not None and conf >= self.cache_accept_conf:
            self.cached_timer_bbox = tuple(int(v) for v in bbox)
        if conf > 0.0:
            self.last_conf = conf

    def reset_cache(self) -> None:
        self.cached_timer_bbox = None
        self.last_conf = None

    def infer(self, frame_bgr: np.ndarray) -> Dict:
        used_cache = False
        result = None

        if self.cached_timer_bbox is not None:
            used_cache = True
            result = infer_timer_from_bbox(
                frame_bgr=frame_bgr,
                model=self.model,
                layout=self.layout,
                timer_xyxy=self.cached_timer_bbox,
                device=self.device,
            )
            refresh_needed = (
                result.get("text") is None
                or float(result.get("conf", 0.0) or 0.0) < self.refresh_conf_threshold
                or (
                    self.last_conf is not None
                    and self.last_conf - float(result.get("conf", 0.0) or 0.0) >= self.conf_drop_delta
                )
            )
            if refresh_needed:
                refreshed = infer_timer(
                    frame_bgr=frame_bgr,
                    model=self.model,
                    layout=self.layout,
                    device=self.device,
                    anchor_template=self.anchor_template,
                )
                refreshed_conf = float(refreshed.get("conf", 0.0) or 0.0)
                cached_conf = float(result.get("conf", 0.0) or 0.0)
                if refreshed.get("timer_bbox") is not None and refreshed_conf >= cached_conf:
                    result = refreshed
                    used_cache = False
        if result is None:
            result = infer_timer(
                frame_bgr=frame_bgr,
                model=self.model,
                layout=self.layout,
                device=self.device,
                anchor_template=self.anchor_template,
            )
            used_cache = False

        self._update_cache(result)
        result["timer_source"] = "cached_bbox" if used_cache else "anchor_refresh"
        return result


def fmt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def frame_step_from_sample_fps(video_fps: float, sample_fps: float) -> int:
    sample_fps = max(sample_fps, 1e-6)
    return max(1, int(round(video_fps / sample_fps)))


def iter_sampled_frames(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    frame_step: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            return
        yield frame_idx, frame

        next_frame = frame_idx + frame_step
        if next_frame > end_frame:
            return

        grab_count = next_frame - frame_idx - 1
        for _ in range(grab_count):
            if not cap.grab():
                return
        frame_idx = next_frame


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def is_near_zero_timer(text: Optional[str], seconds: Optional[float], max_seconds: float) -> bool:
    if text is None or seconds is None:
        return False
    return text.startswith("00:0") and 0.0 <= seconds < max_seconds


def segment_runs(
    cap: cv2.VideoCapture,
    total_frames: int,
    fps: float,
    tracker: TimerTracker,
    sample_fps: float,
    min_timer_conf: float,
    near_zero_max_seconds: float,
    min_run_gap_seconds: float,
) -> List[RunWindow]:
    step = frame_step_from_sample_fps(fps, sample_fps)
    n_samples = max(1, ((total_frames - 1) // step) + 1)
    start_frames: List[int] = []
    last_start_frame = -10**12
    in_near_zero = False

    pbar = tqdm(total=n_samples, desc="Segmenting runs", unit="sample")
    for frame_idx, frame in iter_sampled_frames(cap, 0, total_frames - 1, step):
        timer = tracker.infer(frame)
        conf = float(timer.get("conf", 0.0) or 0.0)
        near_zero = conf >= min_timer_conf and is_near_zero_timer(
            text=timer.get("text"),
            seconds=timer.get("seconds"),
            max_seconds=near_zero_max_seconds,
        )

        if near_zero and not in_near_zero:
            gap_seconds = (frame_idx - last_start_frame) / fps if last_start_frame >= 0 else None
            if gap_seconds is None or gap_seconds >= min_run_gap_seconds:
                start_frames.append(frame_idx)
                last_start_frame = frame_idx
        in_near_zero = near_zero
        pbar.update(1)

    pbar.close()

    runs: List[RunWindow] = []
    for i, start_frame in enumerate(start_frames):
        end_frame = total_frames - 1
        if i + 1 < len(start_frames):
            end_frame = start_frames[i + 1] - 1
        runs.append(RunWindow(run_idx=i + 1, start_frame=start_frame, end_frame=end_frame))
    return runs


def toast_match_in_frame(
    frame_bgr: np.ndarray,
    template: ToastTemplate,
    min_score: float,
    lower_green: np.ndarray,
    upper_green: np.ndarray,
    min_green_area: int,
) -> Optional[ToastMatchResult]:
    roi_bgr, _ = crop_toast_roi(frame_bgr)
    green_crop, meta = green_text_crop(
        roi_bgr,
        min_area=min_green_area,
        lower_green=lower_green,
        upper_green=upper_green,
    )
    if green_crop is None:
        return None

    roi_gray = cv2.cvtColor(green_crop, cv2.COLOR_BGR2GRAY)
    return match_single_template(
        roi_gray=roi_gray,
        template=template,
        min_score=min_score,
        scales=MATCH_SCALES,
    )


def refine_first_match(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    template: ToastTemplate,
    min_score: float,
    lower_green: np.ndarray,
    upper_green: np.ndarray,
    min_green_area: int,
) -> Optional[Tuple[int, np.ndarray, ToastMatchResult]]:
    for frame_idx in range(start_frame, end_frame + 1):
        frame = read_frame_at(cap, frame_idx)
        if frame is None:
            return None
        match = toast_match_in_frame(
            frame_bgr=frame,
            template=template,
            min_score=min_score,
            lower_green=lower_green,
            upper_green=upper_green,
            min_green_area=min_green_area,
        )
        if match is not None:
            return frame_idx, frame, match
    return None


def find_first_toast_in_run(
    cap: cv2.VideoCapture,
    run: RunWindow,
    template: ToastTemplate,
    start_frame: int,
    fps: float,
    search_fps: float,
    min_score: float,
    lower_green: np.ndarray,
    upper_green: np.ndarray,
    min_green_area: int,
) -> Optional[Tuple[int, np.ndarray, ToastMatchResult]]:
    step = frame_step_from_sample_fps(fps, search_fps)
    if start_frame > run.end_frame:
        return None

    n_samples = max(1, ((run.end_frame - start_frame) // step) + 1)
    desc = f"Run {run.run_idx} {template.name}"
    pbar = tqdm(total=n_samples, desc=desc, unit="sample", leave=False)

    for frame_idx, frame in iter_sampled_frames(cap, start_frame, run.end_frame, step):
        match = toast_match_in_frame(
            frame_bgr=frame,
            template=template,
            min_score=min_score,
            lower_green=lower_green,
            upper_green=upper_green,
            min_green_area=min_green_area,
        )
        pbar.update(1)
        if match is None:
            continue

        refine_start = max(start_frame, frame_idx - step + 1)
        refined = refine_first_match(
            cap=cap,
            start_frame=refine_start,
            end_frame=frame_idx,
            template=template,
            min_score=min_score,
            lower_green=lower_green,
            upper_green=upper_green,
            min_green_area=min_green_area,
        )
        pbar.close()
        return refined

    pbar.close()
    return None


def detect_toasts_for_runs(
    cap: cv2.VideoCapture,
    runs: List[RunWindow],
    fps: float,
    tracker: TimerTracker,
    templates: List[ToastTemplate],
    search_fps: float,
    min_score: float,
    lower_green: np.ndarray,
    upper_green: np.ndarray,
    min_green_area: int,
) -> List[PipelineEvent]:
    events: List[PipelineEvent] = []

    for run in runs:
        tracker.reset_cache()
        search_start = run.start_frame
        for toast_index, template in enumerate(templates, start=1):
            found = find_first_toast_in_run(
                cap=cap,
                run=run,
                template=template,
                start_frame=search_start,
                fps=fps,
                search_fps=search_fps,
                min_score=min_score,
                lower_green=lower_green,
                upper_green=upper_green,
                min_green_area=min_green_area,
            )
            if found is None:
                break

            frame_idx, frame, match = found
            timer = tracker.infer(frame)
            seconds = frame_idx / fps
            events.append(
                PipelineEvent(
                    run_idx=run.run_idx,
                    toast_index=toast_index,
                    timestamp=fmt_timestamp(seconds),
                    frame=frame_idx,
                    timer_text=timer.get("text") or "",
                    timer_conf=float(timer.get("conf", 0.0) or 0.0),
                    template_name=template.name,
                    template_score=float(match.score),
                    timer_source=str(timer.get("timer_source", "")),
                )
            )
            search_start = frame_idx + 1

    return events


def write_output(
    csv_path: str,
    summary_path: str,
    video_path: str,
    fps: float,
    total_frames: int,
    runs: List[RunWindow],
    events: List[PipelineEvent],
    args: argparse.Namespace,
) -> None:
    duration_s = total_frames / fps
    events_by_run: Dict[int, List[PipelineEvent]] = {}
    for event in events:
        events_by_run.setdefault(event.run_idx, []).append(event)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run",
            "run_start_timestamp",
            "run_start_frame",
            "run_end_timestamp",
            "run_end_frame",
            "toast_index",
            "toast_timestamp",
            "toast_frame",
            "timer_text",
            "timer_conf",
            "toast_conf",
            "template_name",
            "timer_source",
        ])

        for run in runs:
            run_start_s = run.start_frame / fps
            run_end_s = run.end_frame / fps
            run_events = sorted(events_by_run.get(run.run_idx, []), key=lambda e: e.toast_index)
            if not run_events:
                writer.writerow([
                    run.run_idx,
                    fmt_timestamp(run_start_s),
                    run.start_frame,
                    fmt_timestamp(run_end_s),
                    run.end_frame,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ])
                continue

            for event in run_events:
                writer.writerow([
                    run.run_idx,
                    fmt_timestamp(run_start_s),
                    run.start_frame,
                    fmt_timestamp(run_end_s),
                    run.end_frame,
                    event.toast_index,
                    event.timestamp,
                    event.frame,
                    event.timer_text or "",
                    fmt_optional_float(event.timer_conf),
                    fmt_optional_float(event.template_score),
                    event.template_name,
                    event.timer_source,
                ])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Pipeline output for {video_path}\n")
        f.write(f"# segmentation_fps={args.segmentation_fps}  toast_search_fps={args.toast_search_fps}\n")
        f.write(f"# min_score={args.min_score}  near_zero_max_seconds={args.near_zero_max_seconds}\n")
        f.write(f"# fps={fps}  total_frames={total_frames}  duration={duration_s:.0f}s\n")
        f.write(f"# runs={len(runs)}  runs_with_toasts={len(events_by_run)}  events={len(events)}\n\n")

        for run in runs:
            run_events = sorted(events_by_run.get(run.run_idx, []), key=lambda e: e.toast_index)
            if not run_events:
                continue

            run_start_s = run.start_frame / fps
            run_end_s = run.end_frame / fps
            f.write(
                f"run {run.run_idx}: "
                f"start={fmt_timestamp(run_start_s)} frame={run.start_frame}  "
                f"end={fmt_timestamp(run_end_s)} frame={run.end_frame}\n"
            )

            for event in run_events:
                f.write(
                    f"  toast{event.toast_index}: "
                    f"timestamp={event.timestamp} frame={event.frame}  "
                    f"timer={event.timer_text or ''}  "
                    f"timer_conf={fmt_optional_float(event.timer_conf)}  "
                    f"toast_conf={fmt_optional_float(event.template_score)}  "
                    f"template={event.template_name}  "
                    f"timer_source={event.timer_source}\n"
                )
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run-aware VOD analysis pipeline")
    parser.add_argument("--video", required=True, help="Path to the VOD mp4")
    parser.add_argument("--toast_templates", default="templates/toast_templates",
                        help="Directory containing ordered toast template PNGs")
    parser.add_argument("--timer_model", default="timer_model.pth",
                        help="Path to trained TinyCharCNN weights")
    parser.add_argument("--timer_layout", default="configs/timer_layout_from_anchor.json",
                        help="Path to timer layout config")
    parser.add_argument("--hsv_config", default="configs/hsv_config.json",
                        help="Path to HSV config for green toast filtering")
    parser.add_argument("--min_score", type=float, default=0.55,
                        help="Minimum template match score")
    parser.add_argument("--segmentation_fps", type=float, default=0.5,
                        help="Sample rate used to find run starts")
    parser.add_argument("--toast_search_fps", type=float, default=5.0,
                        help="Sample rate used for ordered toast search within each run")
    parser.add_argument("--near_zero_max_seconds", type=float, default=10.0,
                        help="Treat timer readings below this as run starts")
    parser.add_argument("--min_run_gap_seconds", type=float, default=20.0,
                        help="Minimum allowed gap between detected run starts")
    parser.add_argument("--timer_conf_min", type=float, default=0.50,
                        help="Minimum timer confidence for run segmentation")
    parser.add_argument("--refresh_conf_threshold", type=float, default=0.70,
                        help="Refresh anchor if cached timer confidence drops below this")
    parser.add_argument("--cache_accept_conf", type=float, default=0.80,
                        help="Only cache timer bbox when confidence exceeds this")
    parser.add_argument("--conf_drop_delta", type=float, default=0.30,
                        help="Refresh anchor if cached timer confidence suddenly drops by this amount")
    parser.add_argument("--min_green_area", type=int, default=150,
                        help="Minimum connected green area before attempting template matching")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None,
                        help="Output base path without extension (default: <video>_pipeline)")
    args = parser.parse_args()

    templates = load_toast_templates(args.toast_templates, scales=MATCH_SCALES)
    if not templates:
        raise FileNotFoundError(f"No templates found in {args.toast_templates}")
    print(f"Loaded ordered toast templates: {[t.name for t in templates]}")

    layout = load_timer_layout(args.timer_layout)
    timer_model = TinyCharCNN()
    timer_model.load_state_dict_compat(
        torch.load(args.timer_model, map_location=args.device, weights_only=True)
    )
    timer_model.to(args.device).eval()

    anchor_tmpl = cv2.imread(layout.anchor_template_path, cv2.IMREAD_COLOR)
    if anchor_tmpl is None:
        raise FileNotFoundError(f"Anchor template not found: {layout.anchor_template_path}")

    lower_green, upper_green = load_hsv_config(args.hsv_config)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_base = Path(args.out) if args.out else Path(Path(args.video).stem + "_pipeline")
    csv_path = str(out_base.with_suffix(".csv"))
    summary_path = str(out_base.with_suffix(".txt"))

    tracker = TimerTracker(
        model=timer_model,
        layout=layout,
        device=args.device,
        anchor_template=anchor_tmpl,
        refresh_conf_threshold=args.refresh_conf_threshold,
        conf_drop_delta=args.conf_drop_delta,
        cache_accept_conf=args.cache_accept_conf,
    )

    runs = segment_runs(
        cap=cap,
        total_frames=total_frames,
        fps=fps,
        tracker=tracker,
        sample_fps=args.segmentation_fps,
        min_timer_conf=args.timer_conf_min,
        near_zero_max_seconds=args.near_zero_max_seconds,
        min_run_gap_seconds=args.min_run_gap_seconds,
    )
    print(f"Detected {len(runs)} run(s)")

    events = detect_toasts_for_runs(
        cap=cap,
        runs=runs,
        fps=fps,
        tracker=tracker,
        templates=templates,
        search_fps=args.toast_search_fps,
        min_score=args.min_score,
        lower_green=lower_green,
        upper_green=upper_green,
        min_green_area=args.min_green_area,
    )

    cap.release()
    write_output(
        csv_path=csv_path,
        summary_path=summary_path,
        video_path=args.video,
        fps=fps,
        total_frames=total_frames,
        runs=runs,
        events=events,
        args=args,
    )
    print(f"{len(events)} event(s) written to {summary_path}")
    print(f"dataset written to {csv_path}")


if __name__ == "__main__":
    main()
