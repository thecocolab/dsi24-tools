from typing import Optional, Tuple, Dict
import numpy as np
import mne
from scipy.signal import welch
from antropy import lziv_complexity
from manager import Processor


def compute_spectrum(x: np.ndarray, info: mne.Info, result: Dict[str, np.ndarray]):
    nperseg = min(256, x.shape[1])
    result["freq"], result["spec"] = welch(x, info["sfreq"], nperseg=nperseg)


class PSD(Processor):
    band_mapping: Dict[str, Tuple[float, float]] = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 50),
    }

    def __init__(
        self, name: str, fmin: Optional[float] = None, fmax: Optional[float] = None
    ):
        self.name = name

        if name in self.band_mapping:
            fmin_default, fmax_default = self.band_mapping[name]
            if fmin is None:
                fmin = fmin_default
            if fmax is None:
                fmax = fmax_default
        elif fmin is None or fmax is None:
            raise RuntimeError(
                f"If name ({name}) is not part of the built-in bands "
                f"({', '.join(self.band_mapping.keys())}), "
                f"fmin ({fmin}) and fmax ({fmax}) can't be None."
            )
        self.fmin = fmin
        self.fmax = fmax

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, np.ndarray],
        intermediates: Dict[str, np.ndarray],
    ):
        if "spec" not in intermediates:
            compute_spectrum(raw, info, intermediates)

        mask = intermediates["freq"] >= self.fmin
        mask &= intermediates["freq"] < self.fmax
        if mask.any():
            processed[self.name] = intermediates["spec"][:, mask].mean(axis=1)
        else:
            processed[self.name] = np.zeros(raw.shape[0])


class LempelZiv(Processor):
    def __init__(self, binarize_mode: str = "mean"):
        assert binarize_mode in [
            "mean",
            "median",
        ], "binarize_mode should be either mean or median"
        self.binarize_mode = binarize_mode

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, np.ndarray],
        intermediates: Dict[str, np.ndarray],
    ):
        if self.binarize_mode == "mean":
            binarized = raw >= np.mean(raw, axis=-1, keepdims=True)
        else:
            binarized = raw >= np.median(raw, axis=-1, keepdims=True)
        processed["lempel-ziv"] = np.array(
            [lziv_complexity(ch, normalize=True) for ch in binarized]
        )
