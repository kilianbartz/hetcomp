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
EVERY_N_BLOCK = 2


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
    stats_list = []

    # Laden der Audiodatei
    y, sr = librosa.load(audio_file, sr=None)

    results = [
        analyze_audio_block((y[i : i + BLOCKSIZE], i))
        for i in tqdm(range(0, len(y) - BLOCKSIZE, EVERY_N_BLOCK))
    ]

    # Filter out None results
    stats_list = [stat for stat in results if stat is not None]

    # write stats to csv
    df = pd.DataFrame(stats_list)
    df.to_csv("statistics.csv", index=False)
    print(f"Memory usage after function: {memory_usage()} bytes")  # free memory
    del y
    del results
    del stats_list
    del df
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
