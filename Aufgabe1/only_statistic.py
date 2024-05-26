import numpy as np
import librosa
from multiprocessing import Pool
import pandas as pd
import psutil
import os
import threading
import time
import csv
from tqdm import tqdm
import gc

stop_mem_recording = False


def memory_usage() -> tuple:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


BLOCKSIZE = 2048
MEMORY_SAMPLING_INTERVAL = 3
NUM_SLICES = 2


def record_memory_usage():
    global stop_mem_recording
    with open(f"memory_usage_{int(time.time())}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Time (s)",
                "Memory Usage [PSUtil] (bytes)",
            ]
        )
        start_time = time.time()
        while not stop_mem_recording:
            elapsed_time = time.time() - start_time
            memory = memory_usage()
            writer.writerow([elapsed_time, memory])
            time.sleep(MEMORY_SAMPLING_INTERVAL)


def analyze_audio_block(block):

    # Get the current block
    y_block, idx = block

    if len(y_block) < BLOCKSIZE:
        # Letzter Block könnte kürzer als die Blockgröße sein
        return None

    # Berechnung der Fourier-Transformierten
    Y = np.fft.fft(y_block)
    Y_magnitude = np.abs(Y)[: BLOCKSIZE // 2]

    # Berechnung der statistischen Werte
    mean_val = np.mean(Y_magnitude)
    std_val = np.std(Y_magnitude)
    quantiles = np.percentile(Y_magnitude, [25, 50, 75])

    # Speichern der Statistiken
    stats = {
        "block_start": idx,
        "mean": mean_val,
        "std": std_val,
        "25th_percentile": quantiles[0],
        "50th_percentile": quantiles[1],
        "75th_percentile": quantiles[2],
        "freq_with_highest_amplitude": np.argmax(Y_magnitude),
        "amplitude_of_freq_with_highest_amplitude": np.max(Y_magnitude),
    }

    return stats


def analyze_audio_blocks(audio_file):
    print(f"Initial memory usage: {memory_usage()} bytes")
    # Initialisierung der Liste zur Speicherung der Statistiken
    with open("statistics.csv", "w") as f:
        f.write(
            "block_start,mean,std,25th_percentile,50th_percentile,75th_percentile,freq_with_highest_amplitude,amplitude_of_freq_with_highest_amplitude\n"
        )

    # Laden der Audiodatei
    y, sr = librosa.load(audio_file, sr=None)
    slice_size = (len(y) - BLOCKSIZE) // NUM_SLICES
    slice_indices = [(i * slice_size, (i + 1) * slice_size) for i in range(NUM_SLICES)]
    slice_indices[-1] = (slice_indices[-1][0], len(y) - BLOCKSIZE)
    for j in range(NUM_SLICES):
        stats_list = []
        for i in tqdm(range(slice_indices[j][0], slice_indices[j][1])):
            stats = analyze_audio_block((y[i : i + BLOCKSIZE], i))
            if stats is not None:
                stats_list.append(stats)
        with open("statistics.csv", "a") as f:
            for stats in stats_list:
                f.write(
                    f"{stats['block_start']},{stats['mean']},{stats['std']},{stats['25th_percentile']},{stats['50th_percentile']},{stats['75th_percentile']},{stats['freq_with_highest_amplitude']},{stats['amplitude_of_freq_with_highest_amplitude']}\n"
                )
    del y
    gc.collect()


# Rest of the code remains the same
if __name__ == "__main__":
    for i in range(3):
        stop_mem_recording = False
        audio_file = "nicht_zu_laut_abspielen.wav"

        # Start memory recording thread
        memory_thread = threading.Thread(target=record_memory_usage)
        memory_thread.start()

        analyze_audio_blocks(audio_file)
        stop_mem_recording = True
        # Stop memory recording thread
        memory_thread.join()
