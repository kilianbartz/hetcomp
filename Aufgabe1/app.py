import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pydub import AudioSegment
from scipy.fftpack import fft
from scipy.io import wavfile
import mplcursors
import math

BLOCKSIZE = 0.05


class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")

        self.audio_segment = None
        self.sample_rate = None
        self.samples = None

        self.create_widgets()
        self.plot_figures()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.cursor = None

    def test(self, *args, **kwargs):
        print("args", args)
        print("kwargs", kwargs)

    def create_widgets(self):
        self.file_button = tk.Button(
            self.root, text="Open Audio File", command=self.load_file
        )
        self.file_button.pack()

        self.scrollbar = tk.Scrollbar(
            self.root, orient=tk.HORIZONTAL, command=self.update_plots
        )
        self.scrollbar.pack(fill=tk.X)

        self.figure_frame = tk.Frame(self.root)
        self.figure_frame.pack()

        self.left_plot_figure, self.left_plot_ax = plt.subplots()
        self.right_plot_figure, self.right_plot_ax = plt.subplots()

        self.left_canvas = FigureCanvasTkAgg(
            self.left_plot_figure, master=self.figure_frame
        )
        self.left_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.right_canvas = FigureCanvasTkAgg(
            self.right_plot_figure, master=self.figure_frame
        )
        self.right_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            self.audio_segment = AudioSegment.from_file(file_path)
            self.sample_rate, self.samples = wavfile.read(file_path)
            self.samples = self.samples[:, 0]  # Convert to mono if stereo
            self.update_plots()

    def update_plots(self, *args):
        if self.samples is None:
            return
        if args:
            to = float(args[1])
            self.scrollbar.set(to, to + 0.01)

        pos = self.scrollbar.get()[0] * len(self.samples)
        pos = int(pos)

        time_window = BLOCKSIZE  # 50ms window
        sample_window = int(self.sample_rate * time_window)
        start = max(0, pos - sample_window // 2)
        end = min(len(self.samples), start + sample_window)

        selected_samples = self.samples[start:end]
        times = np.linspace(
            start / self.sample_rate, end / self.sample_rate, num=len(selected_samples)
        )

        self.left_plot_ax.clear()
        self.left_plot_ax.plot(times, selected_samples)
        self.left_plot_ax.set_title("Waveform")
        self.left_plot_ax.set_xlabel("Time [s]")
        self.left_plot_ax.set_ylabel("Amplitude")

        N = len(selected_samples)
        yf = fft(selected_samples)
        xf = np.linspace(0.0, self.sample_rate / 2.0, N // 2)
        magnitude = 2.0 / N * np.abs(yf[: N // 2])
        magnitude_db = 20 * np.log10(magnitude)

        self.right_plot_ax.clear()
        self.right_plot_ax.plot(xf, magnitude_db)
        self.right_plot_ax.set_title("Frequency Magnitude")
        self.right_plot_ax.set_xlabel("Frequency [Hz]")
        self.right_plot_ax.set_ylabel("Magnitude")

        self.left_canvas.draw()
        self.right_canvas.draw()
        # Add cursor with tooltip for the frequency plot
        if self.cursor:
            self.cursor.remove()
        self.cursor = mplcursors.cursor(self.right_plot_ax, hover=True)

        @self.cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(f"{sel.target[0]:.2f} Hz, {sel.target[1]:.2f} dB")

        # print largest 10 frequencies with their magnitude_db values
        max_magnitudes = np.sort(magnitude_db)[-12:]
        max_indices = np.argsort(magnitude_db)[-12:]
        for i in range(12):
            print(
                f"Frequency: {xf[max_indices[i]]:.2f} Hz, Magnitude: {max_magnitudes[i]:.2f} dB"
            )

    def plot_figures(self):
        self.left_plot_ax.plot()
        self.right_plot_ax.plot()
        self.left_canvas.draw()
        self.right_canvas.draw()

    def on_close(self):
        # Handle the window close event
        self.root.quit()  # Stops the mainloop
        self.root.destroy()  # Destroys the window


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()
