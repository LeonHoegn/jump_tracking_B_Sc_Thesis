import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def get_threshold():
    return 0.2
def get_fps():
    return 30
def get_controll_time():
    return 0.2

def load_pt(path: Path):
    return torch.load(path, map_location="cpu")


def find_first(path: Path, pattern: str) -> Path:
    matches = sorted(path.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Kein Treffer fuer '{pattern}' in {path}")
    return matches[0]


def load_data(pt_file: Path):
    base_dir = pt_file.parent
    data = load_pt(pt_file)
    bbx_path = find_first(base_dir, "bbx.pt")
    bbx = load_pt(bbx_path)
    video_path = find_first(base_dir, "0_input_video.mp4")
    return data, bbx, video_path


def extract_bbx(obj) -> np.ndarray:
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        arr = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else obj
    elif isinstance(obj, dict):
        for k in ["bbx_xyxy", "bbx", "bbox", "bboxes", "boxes"]:
            if k in obj:
                t = obj[k]
                arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
                break
        else:
            raise KeyError("Konnte keine Bounding-Boxes finden. Bitte Key-Pfad nennen.")
    else:
        raise TypeError("Unbekanntes bbx-Format.")

    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 4:
        arr = arr[:, 0, :]
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Unerwartete bbx-Form: {arr.shape}")
    return arr


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


def add_bbx(video_path: Path, bbx_obj, out_path: Path) -> None:
    bbx = extract_bbx(bbx_obj)
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
        if frame_idx < bbx.shape[0]:
            x1, y1, x2, y2 = box_to_xyxy(bbx[frame_idx], w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
    n = int(get_controll_time() * get_fps())

    high_points = np.where((curr > prev) & (curr >= nxt))[0] + 1

    low_points = np.where((curr < prev) & (curr <= nxt))[0] + 1

    if x[0] < x[1]:
        low_points = np.insert(low_points, 0, x[0])

    high_points_over_th = []
    between_low_points = []

    x_mean = np.mean(x)

    for i, hp in enumerate(high_points):
        th = get_threshold()
        diff = x[hp] - x[low_points[i]]
        if (diff >= th) and (x_mean < x[hp] + (0.8 * th)):
            high_points_over_th.append(hp)
            between_low_points.append(low_points[i])
            between_low_points.append(low_points[i+1])

    return np.array(high_points_over_th, dtype=int), np.array(between_low_points, dtype=int)


def main():
    folder = Path("inputs")
    out_root = Path("outputs")
    for pt_file in sorted(folder.rglob("*hmr4d_results.pt")):
        base_dir = pt_file.parent
        has_bbx = any(base_dir.rglob("bbx.pt"))
        has_video = any(base_dir.rglob("0_input_video.mp4"))
        if has_bbx and has_video:
            hmr, bbx, video_path = load_data(pt_file)
            transl = extract_transl(hmr)
            y_smooth = smooth_1d(transl[:, 1], window=11)
            y_peaks, y_between = find_peaks_1d(y_smooth)
            rel_dir = base_dir.relative_to(folder)
            out_dir = out_root / rel_dir
            plot_file = out_dir / f"{pt_file.stem}.png"
            video_file = out_dir / f"{pt_file.stem}.mp4"
            plot_transl(transl, y_smooth, y_peaks, title=pt_file.name, save_path=plot_file)
            add_bbx(video_path, bbx, video_file)


if __name__ == "__main__":
    main()
