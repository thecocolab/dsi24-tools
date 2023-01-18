from typing import Dict, List
import numpy as np
import mne
from abc import ABC, abstractmethod


class DataIn(ABC):
    """
    Abstract data input stream. Derive from this class to implement new input streams.
    """

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
        include_chs (List[str]): list of EEG channels to extract features from
        exclude_chs (List[str]): list of EEG channels to exclude form feature extraction
    """

    def __init__(self, include_chs: List[str] = [], exclude_chs: List[str] = []):
        self.include_chs = include_chs
        self.exclude_chs = exclude_chs

    @abstractmethod
    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
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
        """
        pass

    def __call__(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
        """
        Deriving classes should not override this method. It get's called by the Manager,
        applies channel selection and calles the process method with the channel subset.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations
        """
        if not hasattr(self, "include_chs") or not hasattr(self, "exclude_chs"):
            raise RuntimeError(
                f"Couldn't fine include_chs and/or exclude_chs attributes in {self}, "
                "make sure to call the parent class' __init__ inside the derived Processor."
            )

        # pick channels
        ch_idxs = mne.pick_channels(
            info["ch_names"], self.include_chs, self.exclude_chs
        )
        raw = raw[ch_idxs]
        info = mne.pick_info(info, ch_idxs, copy=True)
        # process the data
        return self.process(
            raw,
            info,
            processed,
            intermediates,
        )


class Normalization(ABC):
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
