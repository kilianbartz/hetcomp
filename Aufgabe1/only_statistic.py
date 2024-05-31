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
import os

stop_mem_recording = False


def memory_usage() -> tuple:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


BLOCKSIZE = 2048
MEMORY_SAMPLING_INTERVAL = 3
NUM_CHUNKS = 2
DB_THRESHOLD = 45


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
    Y_db = librosa.amplitude_to_db(Y_magnitude, ref=np.max)
    major_frequencies = np.where(Y_db > DB_THRESHOLD)[0]

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
        "major_frequenzies": major_frequencies,
    }

    return stats


def analyze_audio_blocks(audio_file):
    print(f"Initial memory usage: {memory_usage()} bytes")
    # empty the file if it already exists
    if os.path.exists("statistics.csv"):
        os.remove("statistics.csv")

    # Laden der Audiodatei
    y, sr = librosa.load(audio_file, sr=None)
    slice_size = (len(y) - BLOCKSIZE) // NUM_CHUNKS
    chunk_indices = [(i * slice_size, (i + 1) * slice_size) for i in range(NUM_CHUNKS)]
    chunk_indices[-1] = (chunk_indices[-1][0], len(y) - BLOCKSIZE)
    for j in range(NUM_CHUNKS):
        stats_list = []
        for i in tqdm(range(chunk_indices[j][0], chunk_indices[j][1])):
            stats = analyze_audio_block((y[i : i + BLOCKSIZE], i))
            if stats is not None:
                stats_list.append(stats)
            break
        df = pd.DataFrame(stats_list)
        df.to_csv("statistics.csv", mode="a", header=False, index=False)
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
