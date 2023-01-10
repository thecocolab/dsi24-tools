from typing import Dict
import numpy as np
import mne
from matplotlib import pyplot as plt
from pythonosc.udp_client import UDPClient
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc import osc_bundle_builder
from manager import DataOut


class PlotRaw(DataOut):
    def __init__(self, scaling=1):
        self.scaling = scaling

        # initialize figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Raw EEG buffer")
        self.ax.set_xlabel("time (s)")
        self.fig.show()
        self.line_plots = None

        self.background_buffer = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.fig_size = self.fig.get_size_inches()

    def update(self, raw: np.ndarray, info: mne.Info, processed: Dict[str, np.ndarray]):
        xs = np.arange(-raw.shape[1], 0) / info["sfreq"]
        raw = raw * 5000 * self.scaling + np.arange(raw.shape[0])[:, None]

        if (self.fig_size != self.fig.get_size_inches()).any():
            # hide lines
            for line in self.line_plots:
                line.set_visible(False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # recapture background
            self.background_buffer = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig_size = self.fig.get_size_inches()

            # unhide lines
            for line in self.line_plots:
                line.set_visible(True)

        # restore background
        self.fig.canvas.restore_region(self.background_buffer)

        # update line plots
        if self.line_plots is None:
            xs = xs[None].repeat(raw.shape[0], axis=0).reshape(raw.shape)
            self.line_plots = self.ax.plot(xs.T, raw.T, c="0", linewidth=0.7)
            self.ax.set_yticks(np.arange(raw.shape[0]), info["ch_names"])
        else:
            for i, line in enumerate(self.line_plots):
                line.set_data(xs, raw[i])
                self.ax.draw_artist(line)

        # rescale axes (might be too slow to call every update)
        self.ax.relim()
        self.ax.autoscale_view()

        # redraw the ax
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()


class PlotProcessed(DataOut):
    def __init__(self):
        # initialize figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Processed EEG features")
        self.ax.set_xlabel("features")
        self.ax.set_ylim(-3, 3)
        self.fig.show()
        self.bar_plots = None

        self.background_buffer = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.fig_size = self.fig.get_size_inches()

    def update(self, raw: np.ndarray, info: mne.Info, processed: Dict[str, np.ndarray]):
        if (self.fig_size != self.fig.get_size_inches()).any():
            # hide bars
            for bar in self.bar_plots:
                bar.set_visible(False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # recapture background
            self.background_buffer = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig_size = self.fig.get_size_inches()

            # unhide bars
            for bar in self.bar_plots:
                bar.set_visible(True)

        # restore background
        self.fig.canvas.restore_region(self.background_buffer)

        # update line plots
        values = [p for p in processed.values()]
        if self.bar_plots is None:
            xs = range(len(processed))
            self.bar_plots = self.ax.bar(
                xs, values, color=[f"C{i}" for i in range(len(processed))]
            )
            self.ax.set_xticks(xs, processed.keys())
            self.fig.canvas.draw()
        else:
            for bar, val in zip(self.bar_plots, values):
                bar.set_height(val)
                self.ax.draw_artist(bar)

            if self.ax.get_xlim()[0] == 0:
                self.ax.relim()
                self.ax.autoscale(axis="x")
                self.fig.canvas.draw()

        # redraw the ax
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()


class OSCStream(DataOut):
    def __init__(self, ip: str, port: int, address_prefix: str = "/"):
        self.address_prefix = address_prefix
        self.client = UDPClient(ip, port)

    def update(self, raw: np.ndarray, info: mne.Info, processed: Dict[str, np.ndarray]):
        bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
        # add individual features as messages to the bundle
        for key, val in processed.items():
            msg = OscMessageBuilder(self.address_prefix + key)
            msg.add_arg(val, OscMessageBuilder.ARG_TYPE_FLOAT)
            bundle.add_content(msg.build())
        # send features packaged into a bundle
        self.client.send(bundle.build())
