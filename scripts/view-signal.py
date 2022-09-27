import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import welch
from mne_realtime import LSLClient


host = "nWlrBbmQBhCDarzO"
sfreq = 300
nchan = 24
buffer_size = sfreq * 5

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

    idx = 0
    while True:
        epoch = next(gen)

        if epoch.size == 0:
            # received no data
            sleep_dur = 0.1
            print(f"buffer empty, waiting {sleep_dur}s")
            time.sleep(sleep_dur)
            continue

        for i, (plot, spec, curr, buff) in enumerate(
            zip(plots, spectrum, epoch, buffers)
        ):
            buff.extend(curr)

            # plot raw signal
            plot.set_data(np.arange(len(buff)) / sfreq, buff - np.mean(buff))

            # plot power spectrum
            freq, amp = welch(buff, sfreq)
            mask = (freq >= 1) & (freq < 90)
            freq, amp = freq[mask], amp[mask]
            amp -= amp.min() - 1e-6
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
