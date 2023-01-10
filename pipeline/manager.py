from typing import List, Dict
import time
from collections import deque
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
    def update(self, raw: np.ndarray, info: mne.Info, processed: Dict[str, float]):
        """
        This function is called by the Manager to send a new batch of data to the output stream.

        Parameters:
            raw (np.ndarray): raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary of extracted normalized features
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


class Manager:
    """
    Central class to manage an EEG input stream, several processing steps,
    feature normalization and multiple output channels.

    Parameters:
        data_in (DataIn): instance of a DataIn stream (e.g. EEGStream, EEGRecording)
        processors (List[Processor]): a list of Processor instances (e.g. PSD, LempelZiv)
        data_out (List[DataOut]): a list of DataOut channels (e.g. OSCStream, PlotProcessed)
        frequency (int): frequency of the data processing loop (-1 to run as fast as possible)
        buffer_seconds (float): size of the internal EEG buffer in seconds
        running_alpha (float): alpha parameter of the running z-transform
    """

    def __init__(
        self,
        data_in: DataIn,
        processors: List[Processor],
        data_out: List[DataOut],
        frequency: int = 10,
        buffer_seconds: float = 5,
        running_alpha: float = 0.999,
    ):
        self.frequency = frequency

        self.data_in = data_in
        self.processors = processors
        self.data_out = data_out

        # initialize running metrics
        self.running_alpha = running_alpha
        self.running_mean = {}
        self.running_var = {}

        # initialize raw buffer
        buffer_size = int(self.data_in.info["sfreq"] * buffer_seconds)
        self.buffer = deque(maxlen=buffer_size)

        self.too_slow_count = 0
        self.step = 0

    def update(self):
        """
        Run a single processing step, which includes the following components:

        1. fetch new data from the input stream and add it to the buffer
        if the buffer is full:
            2. run all processors on the EEG buffer
            3. normalize the computed features according to the normalization strategy
            4. send the raw buffer and normalized features to all output targets
        """
        # fetch raw data
        new_data = self.data_in.receive()
        if new_data is None:
            return

        # update raw buffer
        self.buffer.extend(new_data.T)

        # skip processing and output steps while the buffer is not full
        if len(self.buffer) < self.buffer.maxlen:
            return
        elif self.step == 0:
            print("done")

        # process raw data
        raw = np.array(self.buffer).T
        processed, intermediates = {}, {}
        for processor in self.processors:
            processor(raw, self.data_in.info, processed, intermediates)

        # update running metrics
        # TODO: clean up running metrics, probably best to outsource this code to some class
        alpha = self.running_alpha * min(1, self.step / 100)
        for key, val in processed.items():
            if np.isnan(val):
                val = 0

            if key not in self.running_mean:
                self.running_mean[key] = val
            self.running_mean[key] = alpha * self.running_mean[key] + (1 - alpha) * val

            if key not in self.running_var:
                self.running_var[key] = 1
            self.running_var[key] = (
                alpha * self.running_var[key]
                + (1 - alpha) * (val - self.running_mean[key]) ** 2
            )

        # compute z-transformed features
        result = {}
        for key, val in processed.items():
            result[key] = (val - self.running_mean[key]) / (
                np.sqrt(self.running_var[key]) + 1e-12
            )

        # update data outputs
        for out in self.data_out:
            out.update(raw, self.data_in.info, result)

        self.step += 1

    def run(self):
        """
        Start the fetching and processing loop and limit the loop to run at a
        constant update rate.
        """
        print("Filling buffer...", end="")

        last_time = time.time()
        while True:
            # receive, process and output data
            self.update()

            if self.frequency > 0:
                # ensure a constant sampling frequency
                current_time = time.time()
                sleep_dur = 1 / self.frequency - current_time + last_time
                if sleep_dur >= 0:
                    time.sleep(sleep_dur)
                else:
                    self.too_slow_count += 1
                    print(
                        f"Processing too slow to run at {self.frequency}Hz ({self.too_slow_count})"
                    )
                last_time = time.time()


if __name__ == "__main__":
    import data_in
    import processors
    import data_out

    mngr = Manager(
        data_in=data_in.EEGRecording.make_eegbci(),
        processors=[
            processors.PSD("delta"),
            processors.PSD("theta"),
            processors.PSD("alpha"),
            processors.PSD("beta"),
            processors.PSD("gamma"),
            processors.LempelZiv(),
        ],
        data_out=[data_out.OSCStream("127.0.0.1", 5005), data_out.PlotProcessed()],
    )
    mngr.run()
