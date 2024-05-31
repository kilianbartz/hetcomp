import numpy as np
from scipy.io import wavfile
from multiprocessing import Pool
import pandas as pd
import psutil
import os
import threading
import time
import csv
from tqdm import tqdm
import gc
import os

stop_mem_recording = False


def memory_usage() -> tuple:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


BLOCKSIZE = 2205  # 0.05 seconds
MEMORY_SAMPLING_INTERVAL = 3
NUM_CHUNKS = 2
DB_THRESHOLD = 50
NUM_RUNS = 1


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


def analyze_audio_block(block, sr):

    # Get the current block
    y_block, idx = block

    if len(y_block) < BLOCKSIZE:
        # Letzter Block könnte kürzer als die Blockgröße sein
        return None

    # Berechnung der Fourier-Transformierten
    N = len(y_block)
    yf = np.fft.fft(y_block)
    xf = np.linspace(0.0, sr / 2.0, N // 2)
    magnitude = 2.0 / N * np.abs(yf[: N // 2])
    magnitude_db = 20 * np.log10(magnitude)
    max_magnitudes = np.sort(magnitude_db)[::-1]
    max_indices = np.argsort(magnitude_db)[::-1]
    # to_index should be the first index where the magnitude is smaller than the threshold
    to_index = np.where(max_magnitudes < DB_THRESHOLD)[0][0]
    major_frequencies = [
        # floor the frequency to the nearest integer
        (int(xf[max_indices[i]]), int(magnitude_db[max_indices[i]]))
        for i in range(to_index)
    ]

    # Speichern der Statistiken
    stats = {
        "block_start": idx,
        "block_end": idx + BLOCKSIZE,
        "major_frequencies": major_frequencies,
    }

    return stats


def analyze_audio_blocks(audio_file):
    print(f"Initial memory usage: {memory_usage()} bytes")
    # empty the file if it already exists
    if os.path.exists("statistics.csv"):
        os.remove("statistics.csv")

    # Laden der Audiodatei
    sr, y = wavfile.read(audio_file)
    y = y[:, 0]  # Convert to mono if stereo
    slice_size = (len(y) - BLOCKSIZE) // NUM_CHUNKS
    chunk_indices = [(i * slice_size, (i + 1) * slice_size) for i in range(NUM_CHUNKS)]
    chunk_indices[-1] = (chunk_indices[-1][0], len(y) - BLOCKSIZE)
    for j in range(NUM_CHUNKS):
        stats_list = []
        for i in tqdm(range(chunk_indices[j][0], chunk_indices[j][1])):
            stats = analyze_audio_block((y[i : i + BLOCKSIZE], i), sr)
            if stats is not None and len(stats["major_frequencies"]) > 0:
                stats_list.append(stats)
        df = pd.DataFrame(stats_list)
        df.to_csv("statistics.csv", mode="a", header=False, index=False)
        del df
        del stats_list
        gc.collect()
    del y
    gc.collect()


# Rest of the code remains the same
if __name__ == "__main__":
    try:
        for i in range(NUM_RUNS):
            stop_mem_recording = False
            audio_file = "C:\\Users\\kilia\\OneDrive\\Dokumente\\Studium\\HetComp\\Aufgabe1\\nicht_zu_laut_abspielen.wav"

            # Start memory recording thread
            memory_thread = threading.Thread(target=record_memory_usage)
            memory_thread.start()

            analyze_audio_blocks(audio_file)
            stop_mem_recording = True
            # Stop memory recording thread
            memory_thread.join()
    except KeyboardInterrupt:
        stop_mem_recording = True
        memory_thread.join()
        print("Memory recording stopped")
        print("Exiting...")
        exit(0)
