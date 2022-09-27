import sys
import time
from copy import deepcopy
from threading import Thread
import numpy as np
from collections import deque
from matplotlib import pyplot as plt, gridspec
from mne import viz, channels
from mne_realtime import LSLClient
from pyemma.coordinates.transform import TICA


# TODO: describe what these arguments do
# TODO: make these arguments accessible through a command line interface
use_mock_stream = False
tica_lag = 10
history_length = 20
host = "nWlrBbmQBhCDarzO"
sfreq = 300
nchan = 24
epoch_secs = 30

# the COM port to use
if len(sys.argv) > 1:
    # if specified, get the port from the command line
    port = sys.argv[1]
else:
    # default port
    port = "/dev/rfcomm0"


def _draw_topomap(data, info, ax):
    vmin, vmax = data.min(), data.max()
    absmax = max(abs(vmin), abs(vmax))
    ax_img, _ = viz.plot_topomap(
        data, info, vmin=-absmax, vmax=absmax, axes=ax, contours=False, show=False
    )
    return plt.colorbar(ax_img, ax=ax)


def viz_loop():
    global transformed_epochs, position_history

    # set up plot
    plt.ion()

    gs = gridspec.GridSpec(3, 2, width_ratios=[0.8, 0.2])
    fig = plt.figure()

    # add heatmap axis
    tica_ax = fig.add_subplot(gs[:, 0])
    tica_ax.set_facecolor("0")
    tica_ax.set_xlabel("tICA component 1")
    tica_ax.set_ylabel("tICA component 2")

    # add eigenvalue axes
    eigvals_ax = fig.add_subplot(gs[0, 1])
    eigvals_ax.set_title("tICA eigenvalues")
    topo_ax1 = fig.add_subplot(gs[1, 1])
    topo_ax2 = fig.add_subplot(gs[2, 1])

    # initialize artists
    heatmap, cbar1, cbar2 = None, None, None
    curr_pos = tica_ax.plot(
        [0],
        [0],
        c="C0",
        linewidth=3,
    )[0]
    eigvals = eigvals_ax.bar(range(nchan), np.zeros(nchan), width=1)

    # drawing loop
    while True:
        if epoch_idx == 0:
            continue

        if curr_pos is not None:
            # update current position
            try:
                curr_pos.set_data(*tica_model.transform(np.array(position_history)).T)
            except AttributeError:
                pass

        if transformed_epochs is not None:
            # remove previous tICA landscape
            if heatmap is not None:
                heatmap.remove()

            # redraw tICA landscape
            curr = np.concatenate(transformed_epochs, axis=0)
            heatmap = tica_ax.hexbin(*curr.T, bins="log", cmap="inferno", vmin=0.8)
            transformed_epochs = None

            margin = 0.2
            tica_ax.set_xlim(curr[:, 0].min() - margin, curr[:, 0].max() + margin)
            tica_ax.set_ylim(curr[:, 1].min() - margin, curr[:, 1].max() + margin)

            # clear topomap axes
            topo_ax1.cla()
            topo_ax2.cla()
            if cbar1 is not None:
                cbar1.remove()
            if cbar2 is not None:
                cbar2.remove()

            # redraw eigenvalue plots
            for i, val in enumerate(tica_model.eigenvalues):
                eigvals.patches[i].set_height(val)
            eigvals_ax.autoscale()
            eigvals_ax.relim()

            idxs = np.argsort(tica_model.eigenvalues)[::-1][:2]
            eigvecs = tica_model.eigenvectors[:, idxs].T
            cbar1 = _draw_topomap(eigvecs[0], eeg_info, topo_ax1)
            cbar2 = _draw_topomap(eigvecs[1], eeg_info, topo_ax2)
            topo_ax1.set_title("tICA component 1")
            topo_ax2.set_title("tICA component 2")

        # redraw everything
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()


def fit_tica(epoch):
    global tica_model, transformed_epochs
    print("updating tICA model...", end="")

    # update a local copy of the tICA model
    tica_copy = deepcopy(tica_model)
    try:
        tica_copy.partial_fit(epoch)
    except AttributeError:
        tica_copy.fit(epoch)

    # transform the complete current recording
    transformed_epochs = tica_copy.transform(epochs)
    # update the global tICA model
    tica_model = tica_copy
    print("done")


eeg_info = None
#####################################
######## create a mock stream #######
#####################################
if use_mock_stream:
    from mne.datasets import sample
    from mne.io import read_raw_fif
    from mne_realtime import MockLSLStream

    # Load a file to stream raw data
    data_path = sample.data_path()
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    raw = read_raw_fif(raw_fname).load_data().pick("eeg")

    sfreq = raw.info["sfreq"]
    nchan = raw.info["nchan"]

    eeg_info = raw.info
    eeg_info.set_montage(raw.get_montage())

    mock_stream = MockLSLStream(host, raw, "eeg")
    mock_stream.start()
#####################################
#####################################
#####################################

# fixed length epoch buffers
epoch_steps = int(epoch_secs * sfreq)
epoch_buffer = np.zeros((epoch_steps, nchan))
epochs = []
transformed_epochs = None

# collect the few most recent examples for the tICA live viz
position_history = deque(maxlen=history_length)
last_mean, last_std = 0, 1

# start visualization in a separate thread
viz_thread = Thread(target=viz_loop)
viz_thread.start()

# initialize the tICA model
tica_model = TICA(tica_lag, dim=2)
tica_thread = None

epoch_idx = 0
with LSLClient(host=host) as client:
    if eeg_info is None:
        eeg_info = client.info
        # TODO: make sure we have a montage here and if not, use something like the following code
        # eeg_info.set_montage(channels.make_standard_montage("standard_1020"))

    data_gen = client.iter_raw_buffers()
    while True:
        # receive data
        curr = next(data_gen).T

        if curr.size > 0:
            # add most recent sample to history
            position_history.append((curr[-1] - last_mean) / last_std)

        while curr.size > 0:
            # update the current epoch buffer
            chunk_size = min(epoch_steps - epoch_idx, len(curr))

            epoch_buffer[epoch_idx : epoch_idx + chunk_size] = curr[:chunk_size]
            epoch_idx += chunk_size
            curr = curr[chunk_size:]

            if epoch_idx == epoch_steps:
                # current epoch is done, resetting
                epoch_idx = 0

                # apply z-transform to current epoch
                last_mean = epoch_buffer.mean(axis=0)
                last_std = epoch_buffer.std(axis=0)
                epoch = (epoch_buffer - last_mean) / last_std

                # store current epoch
                epochs.append(epoch)
                print(f"finished epoch {len(epochs)}, ", end="")

                # make sure the previous tICA model finished updating
                # this is a sanity check, if we have to wait here something is wrong
                while tica_thread is not None and tica_thread.is_alive():
                    print("waiting for tica thread to finish (fix your code)")
                    time.sleep(0.1)

                # update tICA model in separate thread
                tica_thread = Thread(target=fit_tica, args=(epoch,))
                tica_thread.start()
