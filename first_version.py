import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from pathlib import Path

def get_threshold():
    return 0.2
def get_fps():
    return 30
def get_controll_time():
    return 0.2

def load_pt(path: Path):
    return torch.load(path, map_location="cpu")


def load_smpl(path: Path):
    data = np.load(path, allow_pickle=True)

    # .smpl from PromptHMR is a zip-based numpy container (npz-like)
    if isinstance(data, np.lib.npyio.NpzFile):
        out = {k: data[k] for k in data.files}
        data.close()
        return out

    return data


def load_input_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return load_pt(path)
    if suffix == ".smpl":
        return load_smpl(path)
    raise ValueError(f"Unsupported input file type: {path}")


def load_pkl(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as pickle_error:
        try:
            import joblib  # type: ignore
            return joblib.load(path)
        except Exception as joblib_error:
            raise RuntimeError(
                f"Konnte {path} nicht laden (pickle: {pickle_error}, joblib: {joblib_error})"
            ) from joblib_error


def find_first(path: Path, pattern: str) -> Path:
    matches = sorted(path.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Kein Treffer fuer '{pattern}' in {path}")
    return matches[0]


def find_first_optional(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.rglob(pattern))
    return matches[0] if matches else None


def load_data(input_file: Path):
    base_dir = input_file.parent
    data = load_input_file(input_file)
    bbx_path = find_first_optional(base_dir, "bbx.pt")
    if bbx_path is not None:
        bbx = load_pt(bbx_path)
    else:
        results_path = find_first_optional(base_dir, "results.pkl")
        if results_path is None:
            raise FileNotFoundError(
                f"Keine Bounding-Box Quelle gefunden in {base_dir} (erwartet: bbx.pt oder results.pkl)"
            )
        bbx = load_pkl(results_path)
    if (any(base_dir.rglob("0_input_video.mp4"))):
        video_path = find_first(base_dir, "0_input_video.mp4")
    else:
        video_path = find_first(base_dir, "*.mp4")
    return data, bbx, video_path


def _as_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _merge_tracks_to_frame_boxes(tracks: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    max_frame = int(max(np.max(frames) for frames, _ in tracks))
    boxes_by_frame = [np.empty((0, 4), dtype=float) for _ in range(max_frame + 1)]

    for frames, boxes in tracks:
        for frame_idx, box in zip(frames.astype(int), boxes):
            if frame_idx < 0:
                continue
            boxes_by_frame[frame_idx] = np.vstack([boxes_by_frame[frame_idx], box.reshape(1, 4)])

    return boxes_by_frame


def extract_bbx_frames(obj) -> list[np.ndarray]:
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        arr = _as_numpy(obj)
        if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 4:
            arr = arr[:, 0, :]
        if arr.ndim == 2 and arr.shape[1] == 4:
            return [arr[i : i + 1].astype(float) for i in range(arr.shape[0])]
        if arr.ndim == 3 and arr.shape[2] == 4:
            return [arr[i].astype(float) for i in range(arr.shape[0])]
        raise ValueError(f"Unerwartete bbx-Form: {arr.shape}")

    if isinstance(obj, dict):
        # PromptHMR results.pkl: obj["people"][track_id] -> {"frames": ..., "bboxes": ...}
        if "people" in obj:
            people = obj["people"]
            if isinstance(people, dict):
                people_iter = people.values()
            elif isinstance(people, (list, tuple)):
                people_iter = people
            else:
                raise TypeError(f"Unbekanntes people-Format: {type(people)}")

            tracks = []
            for person in people_iter:
                if not isinstance(person, dict):
                    continue
                if "frames" not in person:
                    continue
                bboxes = None
                for k in ["bboxes", "bbx", "bbox", "boxes"]:
                    if k in person:
                        bboxes = person[k]
                        break
                if bboxes is None:
                    continue
                frames = _as_numpy(person["frames"]).reshape(-1)
                boxes = _as_numpy(bboxes)
                if boxes.ndim == 1 and boxes.shape[0] == 4:
                    boxes = boxes.reshape(1, 4)
                if boxes.ndim != 2 or boxes.shape[1] != 4:
                    continue
                n = min(frames.shape[0], boxes.shape[0])
                if n == 0:
                    continue
                tracks.append((frames[:n].astype(int), boxes[:n].astype(float)))

            if tracks:
                return _merge_tracks_to_frame_boxes(tracks)

            raise KeyError("Konnte in results.pkl keine gueltigen person-bboxes finden.")

        # Einfache Dict-Varianten
        if "frames" in obj:
            for k in ["bboxes", "bbx", "bbox", "boxes"]:
                if k in obj:
                    frames = _as_numpy(obj["frames"]).reshape(-1)
                    boxes = _as_numpy(obj[k])
                    if boxes.ndim == 1 and boxes.shape[0] == 4:
                        boxes = boxes.reshape(1, 4)
                    if boxes.ndim == 2 and boxes.shape[1] == 4:
                        n = min(frames.shape[0], boxes.shape[0])
                        return _merge_tracks_to_frame_boxes(
                            [(frames[:n].astype(int), boxes[:n].astype(float))]
                        )

        for k in ["bbx_xyxy", "bbx", "bbox", "bboxes", "boxes"]:
            if k in obj:
                arr = _as_numpy(obj[k])
                break
        else:
            raise KeyError("Konnte keine Bounding-Boxes finden. Bitte Key-Pfad nennen.")

        if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 4:
            arr = arr[:, 0, :]
        if arr.ndim == 2 and arr.shape[1] == 4:
            return [arr[i : i + 1].astype(float) for i in range(arr.shape[0])]
        if arr.ndim == 3 and arr.shape[2] == 4:
            return [arr[i].astype(float) for i in range(arr.shape[0])]
        raise ValueError(f"Unerwartete bbx-Form: {arr.shape}")

    raise TypeError("Unbekanntes bbx-Format.")


def box_to_xyxy(box: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(float)
    if 0.0 <= x2 <= 1.5 and 0.0 <= y2 <= 1.5 and 0.0 <= x1 <= 1.5 and 0.0 <= y1 <= 1.5:
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + max(0.0, x2)
        y2 = y1 + max(0.0, y2)
    x1i = max(0, min(int(round(x1)), w - 1))
    y1i = max(0, min(int(round(y1)), h - 1))
    x2i = max(0, min(int(round(x2)), w - 1))
    y2i = max(0, min(int(round(y2)), h - 1))
    return x1i, y1i, x2i, y2i


def add_bbx(
    video_path: Path,
    bbx_obj,
    out_path: Path,
    y_between: np.ndarray,
) -> None:
    boxes_by_frame = extract_bbx_frames(bbx_obj)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Konnte Video nicht oeffnen: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(boxes_by_frame):
            frame_boxes = boxes_by_frame[frame_idx]
            color = (0, 255, 0)
            jump = False
            for min_f, max_f in zip(y_between[0::2], y_between[1::2]):
                if min_f < frame_idx < max_f:
                    jump = True
                    break
            if not jump:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            for box_idx in range(frame_boxes.shape[0]):
                x1, y1, x2, y2 = box_to_xyxy(frame_boxes[box_idx], w, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if jump:
                    text_y = max(0, y1 - 10)
                    cv2.putText(
                        frame,
                        "jumping",
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()


def extract_transl(obj):
    """
    Versucht transl (T,3) aus typischen Strukturen zu holen.
    Passe die Key-Pfade an dein Format an, falls nötig.
    """
    # Fall 1: direkt Tensor/Array
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        arr = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else obj
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr

    # Fall 2: dict mit bekannten Keys !!fall bei GVHMR aus hmr4d_results.pt
    if isinstance(obj, dict):
        # häufig: pred["smpl_params_global"]["transl"]
        if "smpl_params_global" in obj and isinstance(obj["smpl_params_global"], dict):
            if "transl" in obj["smpl_params_global"]:
                t = obj["smpl_params_global"]["transl"]
                return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

        # andere Varianten
        for k in ["transl", "translation", "root_transl", "root_translation"]:
            if k in obj:
                t = obj[k]
                return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

        # PromptHMR .smpl format
        if "bodyTranslation" in obj:
            return np.asarray(obj["bodyTranslation"])

    raise KeyError("Konnte keine transl (T,3) finden. Bitte Key-Pfad nennen.")


def plot_transl(
    transl: np.ndarray,
    y_smooth: np.ndarray,
    y_peaks: np.ndarray,
    title: str,
    save_path: Path | None = None,
):
    plt.figure()
    plt.plot(transl[:, 0], label="z")
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


def smooth_1d(x: np.ndarray, window: int = 11) -> np.ndarray:
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x_pad, kernel, mode="valid")


def find_peaks_1d(x: np.ndarray) -> np.ndarray:
    # Lokale Maxima: x[i-1] < x[i] >= x[i+1]
    if x.size < 3:
        return np.array([], dtype=int)
    prev = x[:-2]
    curr = x[1:-1]
    nxt = x[2:]
    high_points = np.where((curr > prev) & (curr >= nxt))[0] + 1

    low_points = np.where((curr < prev) & (curr <= nxt))[0] + 1

    if x[0] < x[1]:
        low_points = np.insert(low_points, 0, 0)

    high_points_over_th = []
    between_low_points = []

    x_mean = np.mean(x)

    for hp in high_points:
        # Need one low point left and one right of the high point.
        right_idx = np.searchsorted(low_points, hp, side="right")
        left_idx = right_idx - 1
        if left_idx < 0 or right_idx >= low_points.shape[0]:
            continue

        left_low = int(low_points[left_idx])
        right_low = int(low_points[right_idx])
        th = get_threshold()
        diff = x[hp] - x[left_low]
        if (diff >= th) and (x_mean < x[hp] + (0.8 * th)):
            high_points_over_th.append(hp)
            between_low_points.append(left_low)
            between_low_points.append(right_low)

    return np.array(high_points_over_th, dtype=int), np.array(between_low_points, dtype=int)


def main():
    folder = Path("inputs")
    out_root = Path("outputs")
    input_files = sorted(folder.rglob("*hmr4d_results.pt")) + sorted(folder.rglob("*.smpl"))
    print(f"{len(input_files)} files loaded")
    for input_file in sorted(input_files):
        base_dir = input_file.parent
        has_bbx = any(base_dir.rglob("bbx.pt")) or any(base_dir.rglob("results.pkl"))
        has_video = any(base_dir.rglob("*.mp4"))
        if has_bbx and has_video:
            hmr, bbx, video_path = load_data(input_file)
            transl = extract_transl(hmr)
            y_smooth = smooth_1d(transl[:, 1], window=11)
            y_peaks, y_between = find_peaks_1d(y_smooth)
            rel_dir = base_dir.relative_to(folder)
            out_dir = out_root / rel_dir
            plot_file = out_dir / f"{input_file.stem}.png"
            video_file = out_dir / f"{input_file.stem}.mp4"
            plot_transl(transl, y_smooth, y_peaks, title=input_file.name, save_path=plot_file)
            add_bbx(video_path, bbx, video_file, y_between)
        elif has_bbx:
            print("no video in ", base_dir)
        else:
            print("no bbx in ", base_dir)


if __name__ == "__main__":
    main()
