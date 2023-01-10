from typing import Optional, Tuple, Dict, List
import numpy as np
import mne
from scipy.signal import welch
from antropy import lziv_complexity
from manager import Processor


def compute_spectrum(x: np.ndarray, info: mne.Info, result: Dict[str, np.ndarray]):
    # grab indices of unprocessed channels
    ch_idxs = [i for i, ch in enumerate(info["ch_names"]) if f"spec-{ch}" not in result]
    if len(ch_idxs) > 0:
        # compute power spectrum for unprocessed channels
        result["freq"], specs = welch(x[ch_idxs], info["sfreq"])
        # save new power spectra in results
        result.update(
            {f"spec-{info['ch_names'][i]}": spec for i, spec in zip(ch_idxs, specs)}
        )


class PSD(Processor):
    band_mapping: Dict[str, Tuple[float, float]] = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 50),
    }

    def __init__(
        self,
        name: str,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        include_chs: List[str] = [],
        exclude_chs: List[str] = [],
    ):
        super(PSD, self).__init__(include_chs, exclude_chs)
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
        # compute power spectral density, skips channels that have been processed already
        compute_spectrum(raw, info, intermediates)

        # extract relevant frequencies
        mask = intermediates["freq"] >= self.fmin
        mask &= intermediates["freq"] < self.fmax
        if mask.any():
            # save mean spectral power across frequency bins and selected channels
            processed[self.name] = np.mean(
                [intermediates[f"spec-{ch}"][mask] for ch in info["ch_names"]]
            )
        else:
            raise RuntimeError(
                f"This frequency band ({self.fmin} - {self.fmax}Hz) has no values inside "
                f"the range of the power spectrum ({intermediates['freq'].min()} - "
                f"{intermediates['freq'].max()}Hz). Consider buffer size and parameters "
                "of the PSD computation."
            )


class LempelZiv(Processor):
    def __init__(
        self,
        binarize_mode: str = "mean",
        include_chs: List[str] = [],
        exclude_chs: List[str] = [],
    ):
        super(LempelZiv, self).__init__(include_chs, exclude_chs)

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
        # binarize raw signal
        if self.binarize_mode == "mean":
            binarized = raw >= np.mean(raw, axis=-1, keepdims=True)
        else:
            binarized = raw >= np.median(raw, axis=-1, keepdims=True)

        # compute Lempel-Ziv complexity
        processed["lempel-ziv"] = np.mean(
            [lziv_complexity(ch, normalize=True) for ch in binarized]
        )
