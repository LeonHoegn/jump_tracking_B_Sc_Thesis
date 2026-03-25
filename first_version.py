from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

DEFAULT_THRESHOLD = 0.2
DEFAULT_FPS = 30
DEFAULT_CONTROL_TIME = 0.2
BOX_KEYS = ("bboxes", "bbx", "bbox", "boxes")
TRANSLATION_KEYS = ("transl", "translation", "root_transl", "root_translation")

TrackData = tuple[int | None, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class SubjectBoundary:
    """Representative subject positions at the start and end of a video segment."""

    first_center: np.ndarray
    last_center: np.ndarray


@dataclass(frozen=True)
class VideoSegment:
    """All data needed to process one input video directory."""

    base_dir: Path
    input_files: list[Path]
    bbx_obj: Any
    video_path: Path
    subject_boundaries: dict[int | None, SubjectBoundary]


def get_threshold() -> float:
    """Return the minimum jump amplitude threshold."""
    return DEFAULT_THRESHOLD


def get_fps() -> int:
    """Return the default fallback frames-per-second value."""
    return DEFAULT_FPS


def get_controll_time() -> float:
    """Return the legacy control time value."""
    return DEFAULT_CONTROL_TIME


def load_pt(path: Path) -> Any:
    """Load a PyTorch file on CPU."""
    return torch.load(path, map_location="cpu")


def load_smpl(path: Path) -> Any:
    """Load a PromptHMR `.smpl` file or any NumPy-backed container."""
    data = np.load(path, allow_pickle=True)

    if isinstance(data, np.lib.npyio.NpzFile):
        loaded = {key: data[key] for key in data.files}
        data.close()
        return loaded

    return data


def load_input_file(path: Path) -> Any:
    """Load a supported model output file."""
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return load_pt(path)
    if suffix == ".smpl":
        return load_smpl(path)
    raise ValueError(f"Unsupported input file type: {path}")


def load_pkl(path: Path) -> Any:
    """Load a pickle file, falling back to joblib when necessary."""
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except Exception as pickle_error:
        try:
            import joblib  # type: ignore

            return joblib.load(path)
        except Exception as joblib_error:
            raise RuntimeError(
                f"Could not load {path} "
                f"(pickle: {pickle_error}, joblib: {joblib_error})"
            ) from joblib_error


def find_first(path: Path, pattern: str) -> Path:
    """Return the first matching file for a glob pattern."""
    matches = sorted(path.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match found for '{pattern}' in {path}")
    return matches[0]


def find_first_optional(path: Path, pattern: str) -> Path | None:
    """Return the first matching file or `None` if no match exists."""
    matches = sorted(path.rglob(pattern))
    return matches[0] if matches else None


def load_data(input_file: Path) -> tuple[Any, Any, Path]:
    """Load a result file together with its bounding boxes and video."""
    base_dir = input_file.parent
    data = load_input_file(input_file)
    bounding_boxes, video_path = load_bbx_and_video(base_dir)
    return data, bounding_boxes, video_path


def load_bbx_and_video(base_dir: Path) -> tuple[Any, Path]:
    """Load bounding box data and the matching input video for a directory."""
    bbx_path = find_first_optional(base_dir, "bbx.pt")
    if bbx_path is not None:
        bounding_boxes = load_pt(bbx_path)
    else:
        results_path = find_first_optional(base_dir, "results.pkl")
        if results_path is None:
            raise FileNotFoundError(
                "No bounding box source found in "
                f"{base_dir} (expected `bbx.pt` or `results.pkl`)."
            )
        bounding_boxes = load_pkl(results_path)

    preferred_video = find_first_optional(base_dir, "0_input_video.mp4")
    video_path = preferred_video if preferred_video is not None else find_first(base_dir, "*.mp4")
    return bounding_boxes, video_path


def as_numpy(value: Any) -> np.ndarray:
    """Convert tensors and array-like inputs to NumPy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_track_id(track_id: Any) -> int | None:
    """Convert track identifiers to plain Python integers."""
    if track_id is None:
        return None
    if isinstance(track_id, torch.Tensor):
        return int(track_id.item())
    if isinstance(track_id, np.ndarray):
        return int(track_id.reshape(-1)[0])
    return int(track_id)


def merge_tracks_to_frame_entries(
    tracks: list[TrackData],
) -> list[list[tuple[int | None, np.ndarray]]]:
    """Convert per-track boxes into a per-frame representation."""
    max_frame = int(max(np.max(frames) for _, frames, _ in tracks))
    entries_by_frame: list[list[tuple[int | None, np.ndarray]]] = [[] for _ in range(max_frame + 1)]

    for track_id, frames, boxes in tracks:
        for frame_index, box in zip(frames.astype(int), boxes):
            if frame_index < 0:
                continue
            entries_by_frame[frame_index].append((track_id, box.astype(float)))

    return entries_by_frame


def array_boxes_to_frame_entries(box_array: np.ndarray) -> list[list[tuple[int | None, np.ndarray]]]:
    """Normalize array-based box layouts into per-frame entries."""
    if box_array.ndim == 3 and box_array.shape[1] == 1 and box_array.shape[2] == 4:
        box_array = box_array[:, 0, :]

    if box_array.ndim == 2 and box_array.shape[1] == 4:
        return [[(0, box_array[index].astype(float))] for index in range(box_array.shape[0])]

    if box_array.ndim == 3 and box_array.shape[2] == 4:
        return [
            [(box_index, box_array[frame_index, box_index].astype(float)) for box_index in range(box_array.shape[1])]
            for frame_index in range(box_array.shape[0])
        ]

    raise ValueError(f"Unexpected bounding box shape: {box_array.shape}")


def extract_people_tracks(people: Any) -> list[TrackData]:
    """Extract tracked people from PromptHMR-style `results.pkl` content."""
    if isinstance(people, dict):
        people_items = list(people.items())
    elif isinstance(people, (list, tuple)):
        people_items = list(enumerate(people, start=1))
    else:
        raise TypeError(f"Unknown `people` format: {type(people)}")

    tracks: list[TrackData] = []
    for default_track_id, person in people_items:
        if not isinstance(person, dict) or "frames" not in person:
            continue

        boxes = None
        for key in BOX_KEYS:
            if key in person:
                boxes = person[key]
                break
        if boxes is None:
            continue

        frames = as_numpy(person["frames"]).reshape(-1)
        box_array = as_numpy(boxes)
        if box_array.ndim == 1 and box_array.shape[0] == 4:
            box_array = box_array.reshape(1, 4)
        if box_array.ndim != 2 or box_array.shape[1] != 4:
            continue

        item_count = min(frames.shape[0], box_array.shape[0])
        if item_count == 0:
            continue

        track_id = normalize_track_id(person.get("track_id", default_track_id))
        tracks.append(
            (
                track_id,
                frames[:item_count].astype(int),
                box_array[:item_count].astype(float),
            )
        )

    return tracks


def extract_bbx_frame_entries(bounding_boxes: Any) -> list[list[tuple[int | None, np.ndarray]]]:
    """Extract bounding boxes into a `frame -> [(track_id, box)]` layout."""
    if isinstance(bounding_boxes, (np.ndarray, torch.Tensor)):
        return array_boxes_to_frame_entries(as_numpy(bounding_boxes))

    if isinstance(bounding_boxes, dict):
        if "people" in bounding_boxes:
            tracks = extract_people_tracks(bounding_boxes["people"])
            if tracks:
                return merge_tracks_to_frame_entries(tracks)
            raise KeyError("Could not find valid person bounding boxes in `results.pkl`.")

        if "frames" in bounding_boxes:
            frames = as_numpy(bounding_boxes["frames"]).reshape(-1)
            for key in BOX_KEYS:
                if key not in bounding_boxes:
                    continue
                box_array = as_numpy(bounding_boxes[key])
                if box_array.ndim == 1 and box_array.shape[0] == 4:
                    box_array = box_array.reshape(1, 4)
                if box_array.ndim == 2 and box_array.shape[1] == 4:
                    item_count = min(frames.shape[0], box_array.shape[0])
                    tracks = [(0, frames[:item_count].astype(int), box_array[:item_count].astype(float))]
                    return merge_tracks_to_frame_entries(tracks)

        for key in ("bbx_xyxy", "bbx", "bbox", "bboxes", "boxes"):
            if key in bounding_boxes:
                return array_boxes_to_frame_entries(as_numpy(bounding_boxes[key]))

        raise KeyError("Could not find bounding boxes. Please specify the correct key path.")

    raise TypeError("Unknown bounding box format.")


def extract_bbx_reference_size(bounding_boxes: Any) -> tuple[int, int] | None:
    """Extract the reference image size used when the boxes were generated."""
    if not isinstance(bounding_boxes, dict):
        return None

    camera = bounding_boxes.get("camera")
    if not isinstance(camera, dict):
        return None

    image_center = camera.get("img_center")
    if image_center is None:
        return None

    center = as_numpy(image_center).reshape(-1)
    if center.size < 2:
        return None

    reference_width = int(round(float(center[0]) * 2.0))
    reference_height = int(round(float(center[1]) * 2.0))
    if reference_width <= 0 or reference_height <= 0:
        return None

    return reference_width, reference_height


def unpack_box_coordinates(box: np.ndarray) -> tuple[float, float, float, float]:
    """Return a box as four float coordinates."""
    box = box.astype(float)
    if box.shape == (2, 2):
        x1, y1 = box[0]
        x2, y2 = box[1]
    elif box.shape == (4,):
        x1, y1, x2, y2 = box
    else:
        raise ValueError(f"Unexpected box shape for coordinate extraction: {box.shape}")

    if x2 <= x1 or y2 <= y1:
        x2 = x1 + max(0.0, x2)
        y2 = y1 + max(0.0, y2)

    return float(x1), float(y1), float(x2), float(y2)


def normalize_box_coordinates(
    box: np.ndarray,
    reference_width: int | None = None,
    reference_height: int | None = None,
) -> tuple[float, float, float, float]:
    """Normalize box coordinates so they are comparable across videos."""
    x1, y1, x2, y2 = unpack_box_coordinates(box)
    if all(0.0 <= value <= 1.5 for value in (x1, y1, x2, y2)):
        return x1, y1, x2, y2

    if reference_width is not None and reference_height is not None:
        return (
            x1 / float(reference_width),
            y1 / float(reference_height),
            x2 / float(reference_width),
            y2 / float(reference_height),
        )

    return x1, y1, x2, y2


def compute_box_center(
    box: np.ndarray,
    reference_width: int | None = None,
    reference_height: int | None = None,
) -> np.ndarray:
    """Compute the center point of a bounding box."""
    x1, y1, x2, y2 = normalize_box_coordinates(box, reference_width, reference_height)
    return np.array(((x1 + x2) / 2.0, (y1 + y2) / 2.0), dtype=float)


def extract_subject_boundaries(bounding_boxes: Any) -> dict[int | None, SubjectBoundary]:
    """Extract start and end centers for each subject in one video segment."""
    reference_size = extract_bbx_reference_size(bounding_boxes)
    reference_width, reference_height = reference_size if reference_size is not None else (None, None)

    if isinstance(bounding_boxes, dict) and "people" in bounding_boxes:
        subject_boundaries: dict[int | None, SubjectBoundary] = {}
        for track_id, frames, boxes in extract_people_tracks(bounding_boxes["people"]):
            if frames.size == 0 or boxes.shape[0] == 0:
                continue
            subject_boundaries[track_id] = SubjectBoundary(
                first_center=compute_box_center(boxes[0], reference_width, reference_height),
                last_center=compute_box_center(boxes[-1], reference_width, reference_height),
            )
        if subject_boundaries:
            return subject_boundaries

    boxes_by_frame = extract_bbx_frame_entries(bounding_boxes)
    first_frame_boxes = next((frame_boxes for frame_boxes in boxes_by_frame if frame_boxes), [])
    last_frame_boxes = next((frame_boxes for frame_boxes in reversed(boxes_by_frame) if frame_boxes), [])
    if not first_frame_boxes or not last_frame_boxes:
        return {}

    last_boxes_by_track = {
        track_id if track_id is not None else fallback_index: box
        for fallback_index, (track_id, box) in enumerate(last_frame_boxes, start=1)
    }

    subject_boundaries: dict[int | None, SubjectBoundary] = {}
    for fallback_index, (track_id, box) in enumerate(first_frame_boxes, start=1):
        subject_id = track_id if track_id is not None else fallback_index
        last_box = last_boxes_by_track.get(subject_id, box)
        subject_boundaries[subject_id] = SubjectBoundary(
            first_center=compute_box_center(box, reference_width, reference_height),
            last_center=compute_box_center(last_box, reference_width, reference_height),
        )

    return subject_boundaries


def box_to_xyxy(
    box: np.ndarray,
    width: int,
    height: int,
    reference_width: int | None = None,
    reference_height: int | None = None,
) -> tuple[int, int, int, int]:
    """Convert a box representation into pixel-based `(x1, y1, x2, y2)` coordinates."""
    x1, y1, x2, y2 = unpack_box_coordinates(box)

    if all(0.0 <= value <= 1.5 for value in (x1, y1, x2, y2)):
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    elif (
        reference_width is not None
        and reference_height is not None
        and (reference_width != width or reference_height != height)
    ):
        scale_x = width / float(reference_width)
        scale_y = height / float(reference_height)
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

    x1_int = max(0, min(int(round(x1)), width - 1))
    y1_int = max(0, min(int(round(y1)), height - 1))
    x2_int = max(0, min(int(round(x2)), width - 1))
    y2_int = max(0, min(int(round(y2)), height - 1))
    return x1_int, y1_int, x2_int, y2_int


def is_jumping_frame(frame_index: int, jump_ranges: np.ndarray | None) -> bool:
    """Return whether a frame lies inside one of the detected jump intervals."""
    if jump_ranges is None:
        return False

    for start_frame, end_frame in zip(jump_ranges[0::2], jump_ranges[1::2]):
        if start_frame < frame_index < end_frame:
            return True

    return False


def solve_min_cost_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Solve a one-to-one minimum-cost assignment for a dense cost matrix."""
    if cost_matrix.size == 0:
        return []

    row_count, column_count = cost_matrix.shape
    transposed = False
    if row_count > column_count:
        cost_matrix = cost_matrix.T
        row_count, column_count = cost_matrix.shape
        transposed = True

    @lru_cache(maxsize=None)
    def solve(row_index: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if row_index == row_count:
            return 0.0, ()

        best_cost = float("inf")
        best_pairs: tuple[tuple[int, int], ...] = ()
        for column_index in range(column_count):
            if used_mask & (1 << column_index):
                continue

            remaining_cost, remaining_pairs = solve(row_index + 1, used_mask | (1 << column_index))
            total_cost = float(cost_matrix[row_index, column_index]) + remaining_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_pairs = ((row_index, column_index),) + remaining_pairs

        return best_cost, best_pairs

    _, assignment = solve(0, 0)
    if transposed:
        return [(column_index, row_index) for row_index, column_index in assignment]
    return list(assignment)


def map_subjects_across_segments(
    segments: list[VideoSegment],
) -> dict[Path, dict[int | None, int]]:
    """Assign stable global subject IDs across all video segments."""
    next_global_subject_id = 1
    remembered_subject_centers: dict[int, np.ndarray] = {}
    mappings_by_segment: dict[Path, dict[int | None, int]] = {}

    for segment in segments:
        local_subject_ids = sorted(
            segment.subject_boundaries,
            key=lambda subject_id: (-1 if subject_id is None else int(subject_id)),
        )
        if not local_subject_ids:
            mappings_by_segment[segment.base_dir] = {}
            continue

        current_centers = np.array(
            [segment.subject_boundaries[subject_id].first_center for subject_id in local_subject_ids],
            dtype=float,
        )
        known_global_ids = sorted(remembered_subject_centers)
        local_to_global: dict[int | None, int] = {}

        if known_global_ids:
            remembered_centers = np.array(
                [remembered_subject_centers[global_id] for global_id in known_global_ids],
                dtype=float,
            )
            cost_matrix = np.linalg.norm(
                current_centers[:, np.newaxis, :] - remembered_centers[np.newaxis, :, :],
                axis=2,
            )
            for local_index, global_index in solve_min_cost_assignment(cost_matrix):
                local_to_global[local_subject_ids[local_index]] = known_global_ids[global_index]

        for local_subject_id in local_subject_ids:
            if local_subject_id in local_to_global:
                continue
            local_to_global[local_subject_id] = next_global_subject_id
            next_global_subject_id += 1

        mappings_by_segment[segment.base_dir] = local_to_global
        for local_subject_id, global_subject_id in local_to_global.items():
            remembered_subject_centers[global_subject_id] = segment.subject_boundaries[
                local_subject_id
            ].last_center

    return mappings_by_segment


def resolve_subject_id(
    local_subject_id: int | None,
    subject_id_mapping: dict[int | None, int] | None,
    fallback_subject_id: int | None = None,
) -> int | None:
    """Resolve a local subject ID to its global subject ID."""
    if subject_id_mapping is None:
        return local_subject_id if local_subject_id is not None else fallback_subject_id

    global_subject_id = subject_id_mapping.get(local_subject_id)
    if global_subject_id is not None:
        return global_subject_id

    if local_subject_id is None and len(subject_id_mapping) == 1:
        return next(iter(subject_id_mapping.values()))

    return local_subject_id if local_subject_id is not None else fallback_subject_id


def add_bbx(
    video_path: Path,
    bbx_obj: Any,
    out_path: Path,
    jump_ranges_by_subject: dict[int | None, np.ndarray],
    subject_id_mapping: dict[int | None, int] | None = None,
) -> None:
    """Render subject bounding boxes and jump labels into a video."""
    boxes_by_frame = extract_bbx_frame_entries(bbx_obj)
    bbx_reference_size = extract_bbx_reference_size(bbx_obj)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or float(get_fps())
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if bbx_reference_size is not None and bbx_reference_size != (frame_width, frame_height):
        print(
            "Scaling bounding boxes from reference size "
            f"{bbx_reference_size} to video size {(frame_width, frame_height)}"
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index < len(boxes_by_frame):
            frame_boxes = boxes_by_frame[frame_index]
            for fallback_subject_index, (track_id, box) in enumerate(frame_boxes, start=1):
                local_subject_id = track_id if track_id is not None else fallback_subject_index
                subject_id = resolve_subject_id(
                    local_subject_id,
                    subject_id_mapping,
                    fallback_subject_index,
                )
                jump_ranges = jump_ranges_by_subject.get(subject_id)

                if jump_ranges is None and subject_id_mapping is not None and None in subject_id_mapping:
                    jump_ranges = jump_ranges_by_subject.get(subject_id_mapping[None])
                elif jump_ranges is None and track_id is None and len(frame_boxes) == 1:
                    jump_ranges = jump_ranges_by_subject.get(None)

                is_jumping_subject = is_jumping_frame(frame_index, jump_ranges)
                color = (0, 0, 255) if is_jumping_subject else (0, 255, 0)
                reference_width, reference_height = (
                    bbx_reference_size if bbx_reference_size is not None else (None, None)
                )
                x1, y1, x2, y2 = box_to_xyxy(
                    box,
                    frame_width,
                    frame_height,
                    reference_width,
                    reference_height,
                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_y = max(20, y1 - 10)
                cv2.putText(
                    frame,
                    f"subject-{subject_id}",
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                if is_jumping_subject:
                    jumping_label_y = min(frame_height - 10, y1 + 25)
                    cv2.putText(
                        frame,
                        "jumping",
                        (x1, jumping_label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()


def concatenate_videos(video_paths: list[Path], out_path: Path) -> None:
    """Concatenate multiple videos into one output video in the given order."""
    if not video_paths:
        raise ValueError("No videos provided for concatenation.")

    first_capture = cv2.VideoCapture(str(video_paths[0]))
    if not first_capture.isOpened():
        first_capture.release()
        raise FileNotFoundError(f"Could not open video for concatenation: {video_paths[0]}")

    output_fps = first_capture.get(cv2.CAP_PROP_FPS) or float(get_fps())
    output_width = int(first_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(first_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_capture.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (output_width, output_height))

    for video_path in video_paths:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            writer.release()
            raise FileNotFoundError(f"Could not open video for concatenation: {video_path}")

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame.shape[1] != output_width or frame.shape[0] != output_height:
                frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

            writer.write(frame)

        capture.release()

    writer.release()


def extract_transl(obj: Any) -> np.ndarray:
    """Extract a translation array with shape `(T, 3)` from common result layouts."""
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        transl = as_numpy(obj)
        if transl.ndim == 2 and transl.shape[1] == 3:
            return transl

    if isinstance(obj, dict):
        global_params = obj.get("smpl_params_global")
        if isinstance(global_params, dict) and "transl" in global_params:
            return as_numpy(global_params["transl"])

        for key in TRANSLATION_KEYS:
            if key in obj:
                return as_numpy(obj[key])

        if "bodyTranslation" in obj:
            return np.asarray(obj["bodyTranslation"])

    raise KeyError("Could not find translation data with shape `(T, 3)`.")


def plot_transl(
    transl: np.ndarray,
    y_smooth: np.ndarray,
    y_peaks: np.ndarray,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Plot raw and smoothed translation curves and optionally save the figure."""
    plt.figure()
    plt.plot(transl[:, 0], label="x")
    plt.plot(y_smooth, label="y (smooth)")
    if y_peaks.size >= 1:
        plt.scatter(y_peaks, y_smooth[y_peaks], color="red", s=20, zorder=3, label="y peaks")
    plt.plot(transl[:, 2], label="z")
    plt.xlabel("Frame")
    plt.ylabel("Translation")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

    plt.show()


def smooth_1d(values: np.ndarray, window: int = 11) -> np.ndarray:
    """Apply edge-padded moving-average smoothing to a 1D signal."""
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1

    padding = window // 2
    padded_values = np.pad(values, (padding, padding), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded_values, kernel, mode="valid")


def find_peaks_1d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect jump peaks and the surrounding low points in a 1D signal."""
    if values.size < 3:
        empty = np.array([], dtype=int)
        return empty, empty

    previous_values = values[:-2]
    current_values = values[1:-1]
    next_values = values[2:]

    high_points = np.where((current_values > previous_values) & (current_values >= next_values))[0] + 1
    low_points = np.where((current_values < previous_values) & (current_values <= next_values))[0] + 1

    if values[0] < values[1]:
        low_points = np.insert(low_points, 0, 0)

    qualifying_high_points: list[int] = []
    surrounding_low_points: list[int] = []
    mean_value = float(np.mean(values))
    threshold = get_threshold()

    for high_point in high_points:
        right_index = int(np.searchsorted(low_points, high_point, side="right"))
        left_index = right_index - 1
        if left_index < 0 or right_index >= low_points.shape[0]:
            continue

        left_low = int(low_points[left_index])
        right_low = int(low_points[right_index])
        amplitude = values[high_point] - values[left_low]

        if amplitude >= threshold and mean_value < values[high_point] + (0.8 * threshold):
            qualifying_high_points.append(int(high_point))
            surrounding_low_points.extend((left_low, right_low))

    return np.array(qualifying_high_points, dtype=int), np.array(surrounding_low_points, dtype=int)


def collect_input_files(folder: Path) -> dict[Path, list[Path]]:
    """Group supported input files by their parent directory."""
    input_files = sorted(folder.rglob("*hmr4d_results.pt")) + sorted(folder.rglob("*.smpl"))
    print(f"{len(input_files)} files loaded")

    grouped_files: dict[Path, list[Path]] = {}
    for input_file in sorted(input_files):
        grouped_files.setdefault(input_file.parent, []).append(input_file)

    return grouped_files


def collect_video_segments(folder: Path) -> list[VideoSegment]:
    """Collect all processable video segments for a folder."""
    input_files_by_dir = collect_input_files(folder)
    segments: list[VideoSegment] = []

    for base_dir, base_input_files in sorted(input_files_by_dir.items()):
        has_bbx = any(base_dir.rglob("bbx.pt")) or any(base_dir.rglob("results.pkl"))
        has_video = any(base_dir.rglob("*.mp4"))

        if not has_bbx:
            print(f"No bounding boxes found in {base_dir}")
            continue
        if not has_video:
            print(f"No video found in {base_dir}")
            continue

        bbx_obj, video_path = load_bbx_and_video(base_dir)
        segments.append(
            VideoSegment(
                base_dir=base_dir,
                input_files=sorted(base_input_files),
                bbx_obj=bbx_obj,
                video_path=video_path,
                subject_boundaries=extract_subject_boundaries(bbx_obj),
            )
        )

    return segments


def validate_segment_videos(segments: list[VideoSegment]) -> None:
    """Ensure every segment video can be opened before tracking starts."""
    for segment in segments:
        capture = cv2.VideoCapture(str(segment.video_path))
        if not capture.isOpened():
            capture.release()
            raise FileNotFoundError(f"Could not open video: {segment.video_path}")
        capture.release()


def parse_subject_id_from_file(input_file: Path) -> int | None:
    """Extract the local subject ID from a file name."""
    if input_file.stem.startswith("subject-"):
        return int(input_file.stem.removeprefix("subject-"))
    return None


def process_subject_file(input_file: Path, out_dir: Path) -> tuple[int | None, np.ndarray]:
    """Process one subject file and return its inferred jump ranges."""
    hmr_output = load_input_file(input_file)
    transl = extract_transl(hmr_output)
    y_smooth = smooth_1d(transl[:, 1], window=11)
    y_peaks, jump_ranges = find_peaks_1d(y_smooth)

    plot_path = out_dir / f"{input_file.stem}.png"
    plot_transl(transl, y_smooth, y_peaks, title=input_file.name, save_path=plot_path)
    return parse_subject_id_from_file(input_file), jump_ranges


def main(input_video: str = "") -> None:
    """Run the jump detection visualization pipeline for the selected input folder."""
    inputs_root = Path("inputs")
    folder = inputs_root if input_video == "" else inputs_root / input_video

    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {folder}")

    output_root = Path("outputs") / "input_video"
    segments = collect_video_segments(folder)
    validate_segment_videos(segments)
    subject_mappings = map_subjects_across_segments(segments)

    jump_ranges_by_segment: dict[Path, dict[int | None, np.ndarray]] = {}
    rendered_video_paths: list[Path] = []

    # Phase 1: track jumps for all segments first.
    for segment in segments:
        relative_dir = segment.base_dir.relative_to(folder)
        out_dir = output_root / relative_dir
        subject_mapping = subject_mappings.get(segment.base_dir, {})
        jump_ranges_by_subject: dict[int | None, np.ndarray] = {}

        for input_file in segment.input_files:
            local_subject_id, jump_ranges = process_subject_file(input_file, out_dir)
            global_subject_id = resolve_subject_id(local_subject_id, subject_mapping)
            if global_subject_id is None:
                continue
            jump_ranges_by_subject[global_subject_id] = jump_ranges

        jump_ranges_by_segment[segment.base_dir] = jump_ranges_by_subject

    # Phase 2: render processed videos with BBX matching.
    for segment in segments:
        relative_dir = segment.base_dir.relative_to(folder)
        out_dir = output_root / relative_dir
        subject_mapping = subject_mappings.get(segment.base_dir, {})
        jump_ranges_by_subject = jump_ranges_by_segment.get(segment.base_dir, {})

        video_file = out_dir / "all_subjects.mp4"
        add_bbx(
            segment.video_path,
            segment.bbx_obj,
            video_file,
            jump_ranges_by_subject,
            subject_id_mapping=subject_mapping,
        )
        rendered_video_paths.append(video_file)
        print(f"Finished {segment.base_dir}")

    merged_video_file = output_root / "all_subjects_merged.mp4"
    concatenate_videos(rendered_video_paths, merged_video_file)
    print(f"Created merged video: {merged_video_file}")


if __name__ == "__main__":
    tyro.cli(main)
