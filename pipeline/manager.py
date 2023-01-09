from typing import List, Dict
import time
from collections import deque
import numpy as np
import mne
from abc import ABC, abstractmethod


class DataIn(ABC):
    @property
    @abstractmethod
    def info(self) -> mne.Info:
        pass

    @abstractmethod
    def receive(self) -> np.ndarray:
        pass


class DataOut(ABC):
    @abstractmethod
    def update(self, raw: np.ndarray, info: mne.Info, processed: Dict[str, np.ndarray]):
        pass


class Processor(ABC):
    @abstractmethod
    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, np.ndarray],
        intermediates: Dict[str, np.ndarray],
    ):
        pass


class Manager:
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

        self.step = 0

    def update(self):
        # fetch raw data
        new_data = self.data_in.receive()
        if new_data is None:
            return

        # update raw buffer
        self.buffer.extend(new_data.T)
        raw = np.array(self.buffer).T

        # process raw data
        processed, intermediates = {}, {}
        for processor in self.processors:
            processor.process(raw, self.data_in.info, processed, intermediates)

        # update running metrics
        # TODO: clean up running metrics, probably best to outsource this code to some class
        alpha = self.running_alpha * min(1, self.step / 100)
        for key, val in processed.items():
            val[np.isnan(val)] = 0

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
        last_time = time.time()
        while True:
            # receive, process and output data
            self.update()

            # ensure a constant sampling frequency
            current_time = time.time()
            sleep_dur = 1 / self.frequency - current_time + last_time
            if sleep_dur >= 0:
                time.sleep(sleep_dur)
            else:
                print(f"Processing too slow to run at {self.frequency}Hz.")
            last_time = time.time()


if __name__ == "__main__":
    from data_in import MockEEGStream
    from processors import PSD, LempelZiv
    from data_out import PlotRaw, PlotProcessed

    mngr = Manager(
        MockEEGStream.make_eegbci(),
        [
            PSD("delta"),
            PSD("theta"),
            PSD("alpha"),
            PSD("beta"),
            PSD("gamma"),
            LempelZiv(),
        ],
        [PlotRaw(), PlotProcessed()],
    )
    mngr.run()
