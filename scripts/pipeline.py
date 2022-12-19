from os import path
import time
import numpy as np
from collections import deque
from functools import partial
import warnings
from scipy.signal import welch
from antropy.entropy import lziv_complexity
from mne_realtime import LSLClient
import mne
import pandas as pd


def feature_extractor(fn):
    def feature_wrapper(raw, freq, amp, info, ch_select, **kwargs):
        idxs = mne.pick_channels(info.ch_names, ch_select)
        raw, amp = raw[idxs], amp[idxs]
        return fn(raw, freq, amp, **kwargs)

    return feature_wrapper


class FileWriter:
    def __init__(self, filename, columns, exclude_cols=[], index_scaling=1):
        assert not path.exists(filename), f"File already exists: {filename}"
        self.filename = filename
        self.column_indices = [
            i for i, col in enumerate(columns) if col not in exclude_cols
        ]
        self.original_columns = columns
        self.columns = [columns[i] for i in self.column_indices]
        self.index_scaling = index_scaling
        self.index = 0

        # create the file and write the header
        pd.DataFrame(columns=self.columns).to_csv(self.filename)

    def write_samples(self, samples, index=None):
        # prepare samples
        samples = np.asarray(samples)
        if samples.ndim == 1:
            samples = samples[None]
        assert samples.ndim == 2, f"Dimension should be 1 or 2, got {samples.ndim}"
        assert samples.shape[1] == len(self.original_columns), (
            f"Second dimension of samples ({samples.shape[1]}) should match "
            f"the number of columns specified ({len(self.columns)})"
        )
        # filter out excluded columns
        samples = samples[:, self.column_indices]

        # use a custom index if specified
        if index is not None:
            self.index = index

        # write samples
        curr_index = np.linspace(
            self.index * self.index_scaling,
            (self.index + samples.shape[0]) * self.index_scaling,
            samples.shape[0],
            endpoint=False,
        )
        pd.DataFrame(data=samples, index=curr_index).to_csv(
            self.filename, mode="a", header=False
        )
        self.index += samples.shape[0]


############################################
############ FEATURE EXTRACTORS ############
############################################


@feature_extractor
def psd(raw, freq, amp, lower, upper):
    return amp[:, (freq > lower) & (freq < upper)].mean()


@feature_extractor
def lzc(raw, freq, amp):
    raw = raw - np.median(raw, axis=1, keepdims=True)
    compl = [lziv_complexity(signal) for signal in raw]
    return np.mean(compl)


features = {
    "alpha": partial(psd, ch_select=["O1", "O2"], lower=8, upper=12),
    "beta": partial(psd, ch_select=["C3", "C4"], lower=12, upper=30),
    "lempel-ziv": partial(lzc, ch_select=["Fp1", "Fp2"]),
}
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

############ PARAMETERS ############
raw_buffer_length = 1500
feature_buffer_length = 100
host = "nWlrBbmQBhCDarzO"
sfreq = 300
feature_sfreq = 10
raw_file = "data/raw.csv"
feature_file = "data/features.csv"
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# make sure not to overwrite previous recordings
assert not path.exists(raw_file), f"Raw file {raw_file} already exists"
assert not path.exists(feature_file), f"Feature file {feature_file} already exists"

# initialize feature buffers
feature_buffers = {ft: deque(maxlen=feature_buffer_length) for ft in features.keys()}

# connect to the EEG headset
with LSLClient(host=host) as client:
    gen = client.iter_raw_buffers()

    # initialize buffers
    buffers = [deque(maxlen=raw_buffer_length) for _ in range(client.info["nchan"])]

    # instantiate file writers
    raw_writer = FileWriter(
        raw_file,
        client.info["ch_names"],
        exclude_cols=["X1", "X2", "X3", "TRG"],
        index_scaling=1 / sfreq,
    )
    feature_writer = FileWriter(
        feature_file, list(features.keys()), index_scaling=1 / sfreq
    )

    # acquisition loop
    last_time = time.time()
    while True:
        epoch = next(gen)

        if epoch.size == 0:
            # received no data
            continue

        # add new data to the buffers
        for curr, buff in zip(epoch, buffers):
            buff.extend(curr)

        # compute power spectrum
        raw = np.array(buffers)
        freq, amp = welch(raw, sfreq)

        # extract features
        result = {}
        for ft, func in features.items():
            curr = func(raw, freq, amp, client.info)
            feature_buffers[ft].append(curr)

            mean, std = np.mean(feature_buffers[ft]), np.std(feature_buffers[ft])
            result[ft] = (curr - mean) / std

        # write data to file
        raw_writer.write_samples(epoch.T)
        feature_writer.write_samples(list(result.values()), index=raw_writer.index - 1)

        # ensure constant feature sampling
        current_time = time.time()
        delta = current_time - last_time
        if delta < (1 / feature_sfreq):
            time.sleep((1 / feature_sfreq) - delta)
        else:
            warnings.warn(f"Processing too slow for {feature_sfreq}Hz sampling.")
        print(
            f"Feature sampling frequency: {1 / (time.time() - last_time):.2f}Hz",
            end="\r",
        )
        last_time = time.time()
