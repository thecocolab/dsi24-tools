import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import welch
from mne_realtime import LSLClient
import mne


def _filter(x, sfreq, freq_range):
    if freq_range is None:
        return x
    freq = np.abs(np.fft.fftfreq(x.shape[0], 1 / sfreq))
    spec = np.fft.fft(x, axis=0)
    spec[(freq < freq_range[0]) | (freq > freq_range[1])] = 0
    return np.fft.ifft(spec, axis=0).real


host = "nWlrBbmQBhCDarzO"
sfreq = 300
nchan = 20
buffer_size = sfreq * 10
exclude = ["X1", "X2", "X3", "TRG"]

# initialize plots
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2)
plots = ax1.plot(np.zeros((sfreq, nchan)))
spectrum = ax2.plot(np.ones((sfreq, nchan)))
ax2.loglog()

# initialize buffers
buffers = [deque(maxlen=int(buffer_size)) for _ in range(nchan)]
with LSLClient(host=host) as client:
    gen = client.iter_raw_buffers()

    ch_idxs = mne.pick_channels(client.info.ch_names, [], exclude=exclude)

    idx = 0
    while True:
        epoch = next(gen)

        if epoch.size == 0:
            # received no data
            sleep_dur = 0.1
            print(f"buffer empty, waiting {sleep_dur}s")
            time.sleep(sleep_dur)
            continue

        epoch = epoch[ch_idxs]

        for i, (plot, spec, curr, buff) in enumerate(
            zip(plots, spectrum, epoch, buffers)
        ):
            buff.extend(curr)

            # plot raw signal
            plot.set_data(np.arange(len(buff)) / sfreq, _filter(np.array(buff) - np.mean(buff), sfreq, (8, 12)))

            # plot power spectrum
            freq, amp = welch(_filter(np.array(buff), sfreq, (8, 12)), sfreq)
            mask = (freq >= 1) & (freq < 90)
            freq, amp = freq[mask], amp[mask]
            spec.set_data(freq, amp)

        # rescale plots
        if idx % 10 == 0:
            ax1.autoscale()
            ax1.relim()
            ax2.autoscale()
            ax2.relim()

        # redraw everything
        fig.canvas.draw()
        fig.canvas.flush_events()
        idx += 1
