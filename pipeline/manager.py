import time
from collections import deque
from typing import List

import numpy as np
from mne.io.base import _get_ch_factors
from utils import DataIn, DataOut, Normalization, Processor


class Manager:
    """
    Central class to manage an EEG input stream, several processing steps,
    feature normalization and multiple output channels.

    Parameters:
        data_in (DataIn): instance of a DataIn stream (e.g. EEGStream, EEGRecording)
        processors (List[Processor]): a list of Processor instances (e.g. PSD, LempelZiv)
        normalization (Normalization): the normalization strategy to apply to the extracted features
        data_out (List[DataOut]): a list of DataOut channels (e.g. OSCStream, PlotProcessed)
        frequency (int): frequency of the data processing loop (-1 to run as fast as possible)
        buffer_seconds (float): size of the internal EEG buffer in seconds
    """

    def __init__(
        self,
        data_in: DataIn,
        processors: List[Processor],
        normalization: Normalization,
        data_out: List[DataOut],
        frequency: int = 10,
        buffer_seconds: float = 5,
    ):
        self.data_in = data_in
        self.processors = processors
        self.normalization = normalization
        self.data_out = data_out

        # initialize raw buffer
        buffer_size = int(self.data_in.info["sfreq"] * buffer_seconds)
        self.buffer = deque(maxlen=buffer_size)

        # auxiliary attributes
        self.frequency = frequency
        self.too_slow_count = 0
        self.filling_buffer = True
        self.n_samples_received = -1
        self.samples_missed_count = 0
        self.unit_conversion = None

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
            self.n_samples_received = -1
            return

        # convert the data into micro Volts
        if self.unit_conversion is None:
            self.unit_conversion = _get_ch_factors(
                self.data_in.info, "uV", np.arange(self.data_in.info["nchan"])
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
            return
        elif self.filling_buffer:
            # done filling the buffer
            print("done")
            self.filling_buffer = False

        # process raw data (feature extraction)
        raw = np.array(self.buffer).T
        processed, intermediates = {}, {}
        for processor in self.processors:
            processor(raw, self.data_in.info, processed, intermediates)

        # normalize extracted features
        self.normalization.normalize(processed)

        # update data outputs
        for out in self.data_out:
            out.update(raw, self.data_in.info, processed, self.n_samples_received)

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
    import data_out
    import normalization
    import processors

    mngr = Manager(
        data_in=data_in.EEGRecording.make_eegbci(),
        processors=[
            processors.PSD(label="delta"),
            processors.PSD(label="theta"),
            processors.PSD(label="alpha"),
            processors.PSD(label="beta"),
            processors.PSD(label="gamma"),
            processors.LempelZiv(),
            processors.Ratio("alpha", "theta", "alpha/theta"),
        ],
        normalization=normalization.WelfordsZTransform(),
        data_out=[
            data_out.OSCStream("127.0.0.1", 5005),
            data_out.PlotProcessed(),
            data_out.ProcessedToFile("test.csv", overwrite=True),
        ],
    )
    mngr.run()
