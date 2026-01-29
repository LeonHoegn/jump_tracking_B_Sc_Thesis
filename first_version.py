import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def get_threshold():
    return 1.5
def get_fps():
    return 30
def get_controll_time():
    return 0.2

def load_pt(path: Path):
    return torch.load(path, map_location="cpu")


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


def plot_transl(transl: np.ndarray, title: str):
    # transl: (T,3)
    y_smooth = smooth_1d(transl[:, 1], window=11)
    y_peaks = find_peaks_1d(y_smooth)
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

    return np.where((curr > prev) & (curr >= nxt))[0] + 1


def main():
    folder = Path("inputs")
    for pt_file in sorted(folder.glob("*.pt")):
        data = load_pt(pt_file)
        transl = extract_transl(data)
        plot_transl(transl, title=pt_file.name)


if __name__ == "__main__":
    main()
