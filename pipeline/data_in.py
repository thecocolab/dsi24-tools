from typing import Optional, Union, List
import numpy as np
from mne_realtime import LSLClient, MockLSLStream
import mne
from mne.io import BaseRaw, read_raw, concatenate_raws
from mne.datasets import eegbci
from utils import DataIn

mne.set_log_level(False)


class EEGStream(DataIn):
    """
    Incoming LSL stream with raw EEG data.

    Parameters:
        host (str): the LSL stream's hostname
        port (int): the LSL stream's port (if None, use the LSLClient's default port)
    """

    def __init__(self, host: str, port: Optional[int] = None):
        # start LSL client
        self.client = LSLClient(host=host, port=port)
        self.client.start()
        self.data_iterator = self.client.iter_raw_buffers()

    @property
    def info(self) -> mne.Info:
        """
        Returns the MNE info object corresponding to this EEG stream
        """
        return self.client.get_measurement_info()

    def receive(self) -> np.ndarray:
        """
        Returns newly acquired samples from the EEG stream as a NumPy array
        with shape (Channels, Time). If there are no new samples None is returned.
        """
        data = next(self.data_iterator)
        if data.size == 0:
            return None
        return data


class EEGRecording(EEGStream):
    """
    Stream previously recorded EEG from a file. The data is loaded from the file and streamed
    to a mock LSL stream, which is then accessed via the EEGStream parent-class.

    Parameters:
        raw (str, BaseRaw): file-name of a raw EEG file or an instance of mne.io.BaseRaw
    """

    def __init__(self, raw: Union[str, BaseRaw]):
        # load raw EEG data
        if not isinstance(raw, BaseRaw):
            raw = read_raw(raw)
        raw.load_data().pick("eeg")

        # start the mock LSL stream to serve the EEG recording
        host = "mock-eeg-stream"
        self.mock_stream = MockLSLStream(host, raw, "eeg")
        self.mock_stream.start()

        # start the LSL client
        super(EEGRecording, self).__init__(host=host)

    @staticmethod
    def make_eegbci(
        subjects: Union[int, List[int]] = 1,
        runs: Union[int, List[int]] = [1, 2],
    ):
        """
        Static utility function to instantiate an EEGRecording instance using
        the PhysioNet EEG BCI dataset. This function automatically downloads the
        dataset if it is not present.
        See https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html#mne-datasets-eegbci-load-data
        for information about the dataset and a description of different runs.

        Parameters:
            subjects (int, List[int]): which subject(s) to load data from
            runs (int, List[int]): which run(s) to load from the corresponding subject
        """
        raw = concatenate_raws([read_raw(p) for p in eegbci.load_data(subjects, runs)])
        eegbci.standardize(raw)
        return EEGRecording(raw)
