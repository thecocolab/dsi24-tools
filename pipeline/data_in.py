from typing import Optional, Union, List
import numpy as np
from mne_realtime import LSLClient, MockLSLStream
import mne
from mne.io import BaseRaw, read_raw, concatenate_raws
from mne.datasets import eegbci
from manager import DataIn

mne.set_log_level(False)


class EEGStream(DataIn):
    def __init__(self, host: str, port: Optional[int] = None):
        # start LSL client
        self.client = LSLClient(host=host, port=port)
        self.client.start()
        self.data_iterator = self.client.iter_raw_buffers()

    @property
    def info(self) -> mne.Info:
        return self.client.get_measurement_info()

    def receive(self) -> np.ndarray:
        data = next(self.data_iterator)
        if data.size == 0:
            return None
        return data


class EEGRecording(EEGStream):
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
        raw = concatenate_raws([read_raw(p) for p in eegbci.load_data(subjects, runs)])
        eegbci.standardize(raw)
        return EEGRecording(raw)
