import colorsys
import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Tuple

import mne
import numpy as np
from biotuner.biocolors import audible2visible, scale2freqs, wavelength_to_rgb
from biotuner.biotuner_object import dyad_similarity
from biotuner.metrics import tuning_cons_matrix
from mne.io.base import _get_ch_factors


class DataIn(ABC):
    """
    Abstract data input stream. Derive from this class to implement new input streams.
    """

    buffer = None
    buffer_seconds = None
    n_samples_received = -1
    filling_buffer = True
    samples_missed_count = 0
    unit_conversion = None

    @property
    @abstractmethod
    def info(self) -> mne.Info:
        """
        Implement this property to return the mne.Info object for this input stream.
        """
        pass

    @abstractmethod
    def receive(self) -> np.ndarray:
        """
        This function is called by the Manager to fetch new data.

        Returns:
            data (np.ndarray): an array with newly acquired data samples with shape (Channels, Time)
        """
        pass

    def update(self):
        """
        This function is called by the Manager to update the raw buffer with new data.
        """
        if self.buffer is None:
            # initialize raw buffer
            buffer_size = int(self.info["sfreq"] * self.buffer_seconds)
            self.buffer = deque(maxlen=buffer_size)

        # fetch new data
        new_data = self.receive()
        if new_data is None:
            self.n_samples_received = -1
            return -1

        # convert the data into micro Volts
        if self.unit_conversion is None:
            self.unit_conversion = _get_ch_factors(
                self.info, "uV", np.arange(self.info["nchan"])
            )[:, None]
        new_data *= self.unit_conversion

        # make sure we didn't receive more samples than the buffer can hold
        self.n_samples_received = new_data.shape[1]
        if self.n_samples_received > self.buffer.maxlen:
            self.n_samples_received = self.buffer.maxlen
            self.samples_missed_count += 1
            print(
                f"Received {self.n_samples_received} new samples but the buffer only holds "
                f"{self.buffer.maxlen} samples. Output modules will miss some samples. "
                f"({self.samples_missed_count})"
            )

        # update raw buffer
        self.buffer.extend(new_data.T)

        # skip processing and output steps while the buffer is not full
        if len(self.buffer) < self.buffer.maxlen:
            return -1
        elif self.filling_buffer:
            self.filling_buffer = False
        return self.n_samples_received


class DataOut(ABC):
    """
    Abstract data output stream. Derive from this class to implement new output streams.
    """

    @abstractmethod
    def update(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        n_samples_received: int,
    ):
        """
        This function is called by the Manager to send a new batch of data to the output stream.

        Parameters:
            raw (np.ndarray): raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary of extracted normalized features
            n_samples_received (int): number of new samples in the raw buffer
        """
        pass


class Processor(ABC):
    """
    Abstract data processor. Derive from this class to implement new feature extractors.

    Parameters:
        label (str): the label to be associated with the extracted features
        include_chs (List[str]): list of EEG channels to extract features from
        exclude_chs (List[str]): list of EEG channels to exclude form feature extraction
        normalize (bool): if True, features form this processor will be normalized in the pipeline
    """

    def __init__(
        self,
        label: str,
        include_chs: List[str],
        exclude_chs: List[str],
        normalize: bool = True,
    ):
        self.label = label
        self.include_chs = include_chs
        self.exclude_chs = exclude_chs
        self.normalize = normalize

    @abstractmethod
    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        This function is called internally by __call__ to run the feature extraction.
        Deriving classes should insert the extracted feature into the processed dictionary
        and store intermediate representations that could be useful to other Processors in
        the intermediates dictionary (e.g. the full power spectrum). This function shouldn't
        be called directly as the channel selection is handled by __call__.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations

        Returns:
            features (Dict[str, float]): the extracted features from this processor
        """
        pass

    def __call__(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ) -> Dict[str, bool]:
        """
        Deriving classes should not override this method. It get's called by the Manager,
        applies channel selection and calles the process method with the channel subset.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations

        Returns:
            normalization_mask (Dict[str, bool]): dictionary indicating which features should be normalized
        """
        if not hasattr(self, "include_chs") or not hasattr(self, "exclude_chs"):
            raise RuntimeError(
                f"Couldn't fine include_chs and/or exclude_chs attributes in {self}, "
                "make sure to call the parent class' __init__ inside the derived Processor."
            )
        if self.label in processed:
            raise RuntimeError(
                f'A feature with label "{self.label}" already exists. '
                "Make sure each processor has a unique label."
            )

        # pick channels
        ch_idxs = mne.pick_channels(
            info["ch_names"], self.include_chs, self.exclude_chs
        )
        raw = raw[ch_idxs]
        info = mne.pick_info(info, ch_idxs, copy=True)

        # process the data
        new_features = self.process(
            raw,
            info,
            processed,
            intermediates,
        )
        processed.update(new_features)
        return {lbl: self.normalize for lbl in new_features.keys()}


class Normalization(ABC):
    """
    Abstract normalization class for implementing different normalization strategies.

    Designed to be used with the Processor class, it provides an interface for normalizing
    extracted features. The normalize method applies the strategy, and the reset method
    resets the normalization parameters to their initial state.
    """

    def __init__(self):
        self.user_input_thread = threading.Thread(
            target=self.reset_handler, daemon=True
        )
        self.user_input_thread.start()

    @abstractmethod
    def normalize(self, processed: Dict[str, float]):
        """
        This function is called by the manager to normalize the processed features according
        to the deriving class' normalization strategy. It should modify the processed dictionary
        in-place.

        Parameters:
            processed (Dict[str, float]): dictionary of extracted, unnormalized features
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This function should reset all running or statically acquired normalization parameters
        and reset the normalization to the initial state.
        """
        pass

    def reset_handler(self):
        """
        This function runs in a separate thread to wait for keyboard input, resetting the
        normalization state.
        """
        while True:
            input("Press enter to reset normalization parameters.\n")
            self.reset()


def viz_scale_colors(scale: List[float], fund: float) -> List[Tuple[int, int, int]]:
    """
    Convert a musical scale into a list of HSV colors based on the scale's frequency values
    and their averaged consonance.

    Parameters:
        scale (List[float]): A list of frequency ratios representing the musical scale.
        fund (float): The fundamental frequency of the scale in Hz.

    Returns:
        hsv_all (List[Tuple[float, float, float]]): A list of HSV color tuples, one for each scale step,
        with hue representing the frequency value, saturation representing the consonance, and
        luminance set to a fixed value.
    """

    min_ = 0
    max_ = 1
    # convert the scale to frequency values
    scale_freqs = scale2freqs(scale, fund)
    # compute the averaged consonance of each step
    scale_cons, _ = tuning_cons_matrix(scale, dyad_similarity, ratio_type="all")
    # rescale to match RGB standards (0, 255)
    scale_cons = (np.array(scale_cons) - min_) * (1 / max_ - min_) * 255
    scale_cons = scale_cons.astype("uint8").astype(float) / 255

    hsv_all = []
    for s, cons in zip(scale_freqs, scale_cons):
        # convert freq in nanometer values
        _, _, nm, octave = audible2visible(s)
        # convert to RGB values
        rgb = wavelength_to_rgb(nm)
        # convert to HSV values
        # TODO: colorsys might be slow
        hsv = colorsys.rgb_to_hsv(
            rgb[0] / float(255), rgb[1] / float(255), rgb[2] / float(255)
        )
        hsv = np.array(hsv)
        # rescale
        hsv = (hsv - 0) * (1 / (1 - 0))
        # define the saturation
        hsv[1] = cons
        # define the luminance
        hsv[2] = 200 / 255
        hsv = tuple(hsv)
        hsv_all.append(hsv)

    return hsv_all
