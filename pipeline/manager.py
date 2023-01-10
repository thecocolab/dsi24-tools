from typing import List
import time
from collections import deque
import numpy as np
from utils import DataIn, Processor, DataOut


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
