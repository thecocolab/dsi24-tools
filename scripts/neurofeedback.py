import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import welch
from mne_realtime import LSLClient
import mne


############ EXPERIMENT PARAMETERS ############
include = ["O1", "O2"]  # list of channel names or "all"
exclude = ["X1", "X2", "X3", "TRG"]  # misc channels
buffer_size_seconds = 5
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

############ VARIABLE EXTRACTION FUNCTION ############
extract_variable = lambda freq, amp: amp[(8 < freq) & (freq < 12)].mean()
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

############ CONNECTION PARAMETERS ############
host = "nWlrBbmQBhCDarzO"
sfreq = 300
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# initialize plots
plt.ion()
fig = plt.figure()
bar_plot = plt.bar([0], [1], width=0.4, color="orange")
# clean up the plot
plt.xlim(-0.5, 0.5)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.xticks([])
plt.subplots_adjust(left=0.3, right=0.7)
value_avg = None

# connect to the EEG headset
vmin, vmax = 0, 1
with LSLClient(host=host) as client:
    gen = client.iter_raw_buffers()
    # get relevant channel indices
    ch_idxs = mne.pick_channels(
        client.info.ch_names, [] if include == "all" else include, exclude=exclude
    )

    # initialize buffers
    buffer_size = sfreq * buffer_size_seconds
    buffers = [deque(maxlen=int(buffer_size)) for _ in range(len(ch_idxs))]
    while True:
        epoch = next(gen)

        if epoch.size == 0:
            # received no data
            sleep_dur = 0.1
            print(f"buffer empty, waiting {sleep_dur}s")
            time.sleep(sleep_dur)
            continue

        # select channels
        epoch = epoch[ch_idxs]

        for curr, buff in zip(epoch, buffers):
            buff.extend(curr)

        # compute power spectrum
        freq, amp = welch(np.array(buffers), sfreq)
        amp = amp.mean(axis=0)

        # extract relevant variable from the spectrum
        value = extract_variable(freq, amp)
        if value_avg is None:
            value_avg = value
        else:
            alpha = 0.9
            value_avg = alpha * value_avg + (1 - alpha) * value

        # update axis
        if value_avg < vmin:
            vmin = value_avg
            plt.ylim(vmin, vmax)
        if value_avg > vmax:
            vmax = value_avg
            plt.ylim(vmin, vmax)

        # adjust height of the rect
        next(iter(bar_plot)).set_height(value_avg)

        # redraw everything
        fig.canvas.draw()
        fig.canvas.flush_events()
