import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import welch
from mne_realtime import LSLClient

host = "nWlrBbmQBhCDarzO"
sfreq = 300
nchan = 24
buffer_size = sfreq * 10

plt.ion()
fig, ax = plt.subplots(1)
plots = ax.plot(np.zeros((sfreq, nchan)))
plt.loglog()

buffers = [deque(maxlen=int(buffer_size)) for _ in range(nchan)]

with LSLClient(host=host) as client:
    gen = client.iter_raw_buffers()

    idx = 0
    while True:
        epoch = next(gen)

        if epoch.size == 0:
            sleep_length = 0.1
            print(f"buffer empty, waiting {sleep_length}s")
            time.sleep(sleep_length)
            continue

        for plot, curr, buff in zip(plots, epoch, buffers):
            buff.extend(curr)

            freq, amp = welch(buff, sfreq)

            mask = (freq >= 1) & (freq < 90)
            freq, amp = freq[mask], amp[mask]
            amp -= amp.mean()
            plot.set_data(freq, amp - amp.min() + 1)

        if idx % 10 == 0:
            ax.autoscale()
            ax.relim()
        fig.canvas.draw()
        fig.canvas.flush_events()
        idx += 1
