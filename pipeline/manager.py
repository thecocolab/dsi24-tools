import time
from typing import List

import numpy as np
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
        self.data_in.buffer_seconds = buffer_seconds
        self.processors = processors
        self.normalization = normalization
        self.data_out = data_out

        # auxiliary attributes
        self.frequency = frequency
        self.too_slow_count = 0

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
        if self.data_in.update() == -1:
            return

        # process raw data (feature extraction)
        raw = np.array(self.data_in.buffer).T
        processed, intermediates, normalize_mask = {}, {}, {}
        for processor in self.processors:
            normalize_mask.update(
                processor(raw, self.data_in.info, processed, intermediates)
            )

        # extract the features that need normalization
        finished = {
            lbl: feat for lbl, feat in processed.items() if not normalize_mask[lbl]
        }
        unfinished = {
            lbl: feat for lbl, feat in processed.items() if normalize_mask[lbl]
        }

        # normalize extracted features
        self.normalization.normalize(unfinished)
        finished.update(unfinished)
        finished = {lbl: finished[lbl] for lbl in processed.keys()}

        # update data outputs
        for out in self.data_out:
            out.update(
                raw, self.data_in.info, finished, self.data_in.n_samples_received
            )

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
                        f"Processing too slow to run at {self.frequency}Hz"
                        f" ({self.too_slow_count})"
                    )
                last_time = time.time()


if __name__ == "__main__":
    import data_in
    import data_out
    import normalization
    import processors

    mngr = Manager(
        # data_in=data_in.EEGRecording.make_eegbci(),
        data_in=data_in.EEGStream("Muse00:55:DA:B0:49:D3"),
        processors=[
            processors.PSD(label="delta"),
            processors.PSD(label="theta"),
            processors.PSD(label="alpha"),
            processors.PSD(label="beta"),
            processors.PSD(label="gamma"),
            processors.LempelZiv(),
            processors.Ratio("alpha", "theta", "alpha/theta"),
            processors.Biotuner(),
        ],
        normalization=normalization.StaticBaselineNormal(duration=30),
        data_out=[
            data_out.OSCStream("127.0.0.1", 5005),
            data_out.PlotRaw(),
            data_out.PlotProcessed(),
            # data_out.ProcessedToFile("test.csv", overwrite=True),
        ],
    )
    mngr.run()
