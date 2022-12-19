import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import welch
from mne_realtime import LSLClient
import mne
from pythonosc import udp_client
from antropy.entropy import lziv_complexity


############ EXPERIMENT PARAMETERS ############
include = ["O1", "O2"]  # list of channel names or "all"
exclude = ["X1", "X2", "X3", "TRG"]  # misc channels
buffer_size_seconds = 5
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

############ VARIABLE EXTRACTION FUNCTION ############
extract_variable = lambda freq, amp: amp[(8 < freq) & (freq < 12)].mean()
extract_variable2 = lambda freq, amp: amp[(4 < freq) & (freq < 8)].mean()
extract_variable3 = lambda freq, amp: amp[(12 < freq) & (freq < 30)].mean()
extract_variable4 = lambda x: np.mean(
    [lziv_complexity(cx) for cx in (x - np.median(x, axis=1, keepdims=True))]
)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

############ CONNECTION PARAMETERS ############
host = "nWlrBbmQBhCDarzO"
sfreq = 300
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# initialize plots
plt.ion()
fig = plt.figure()
bar_plot = plt.bar(
    [0, 1, 2, 3], [1, 1, 1, 1], width=0.4, color=["orange", "blue", "red", "green"]
)
# clean up the plot
plt.xlim(-0.5, 3.5)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.xticks([0, 1, 2, 3], ["alpha", "theta", "beta", "complexity"])
plt.subplots_adjust(left=0.3, right=0.7)
value_avg = None
value_avg2 = None
value_avg3 = None
value_avg4 = None

osc_client = udp_client.SimpleUDPClient("192.168.0.196", 5070)

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
            sleep_dur = 0.05
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
        if np.isnan(value):
            continue
        if value_avg is None:
            value_avg = value
        else:
            alpha = 0.9
            value_avg = alpha * value_avg + (1 - alpha) * value

        value2 = extract_variable2(freq, amp)
        if np.isnan(value2):
            continue
        if value_avg2 is None:
            value_avg2 = value2
        else:
            alpha = 0.9
            value_avg2 = alpha * value_avg2 + (1 - alpha) * value2

        value3 = extract_variable3(freq, amp)
        if np.isnan(value3):
            continue
        if value_avg3 is None:
            value_avg3 = value3
        else:
            alpha = 0.9
            value_avg3 = alpha * value_avg3 + (1 - alpha) * value3

        value4 = extract_variable4(np.array(buffers))
        if np.isnan(value4):
            continue
        if value_avg4 is None:
            value_avg4 = value4
        else:
            alpha = 0.9
            value_avg4 = alpha * value_avg4 + (1 - alpha) * value4

        # update axis
        mi = min(value_avg, value_avg2, value_avg3, value_avg4 * 0.014)
        ma = max(value_avg, value_avg2, value_avg3, value_avg4 * 0.014)
        if mi < vmin:
            vmin = mi
            plt.ylim(vmin, vmax)
        if ma > vmax:
            vmax = ma
            plt.ylim(vmin, vmax)

        osc_client.send_message("/baseline/global/spectral_abs/alpha", value_avg)
        osc_client.send_message("/baseline/global/spectral_abs/theta", value_avg2)
        osc_client.send_message("/baseline/global/spectral_abs/beta", value_avg3)
        osc_client.send_message("/baseline/global/spectral_abs/complexity", value_avg4)

        # adjust height of the rect
        it = iter(bar_plot)
        next(it).set_height(value_avg)
        next(it).set_height(value_avg2)
        next(it).set_height(value_avg3)
        next(it).set_height(value_avg4 * 0.014)

        # redraw everything
        fig.canvas.draw()
        fig.canvas.flush_events()
