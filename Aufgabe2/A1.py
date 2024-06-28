import numpy as np
from scipy.io import wavfile
from multiprocessing import Pool, cpu_count
import pandas as pd
import psutil
import os
import threading
import time
import csv
from tqdm import tqdm
import gc

BLOCKSIZE = 2048  # 0.05 seconds
MEMORY_SAMPLING_INTERVAL = 3
DB_THRESHOLD = 50
NUM_RUNS = 1


def analyze_audio_block(block):
    # Get the current block
    y_block, idx, sr = block

    # if len(y_block) < BLOCKSIZE:
    #     # Letzter Block könnte kürzer als die Blockgröße sein
    #     return None

    # Berechnung der Fourier-Transformierten
    N = len(y_block)
    yf = np.fft.fft(y_block)
    xf1 = np.linspace(0.0, sr / 2.0, N // 2)
    xf = np.fft.fftfreq(N, 1 / sr)
    a = list(zip(xf, yf))
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
    start = time.time()
    # initialize csv header
    with open("statistics.csv", mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "block_start",
                "block_end",
                "major_frequencies",
            ]
        )

    # Laden der Audiodatei
    sr, y = wavfile.read(audio_file)
    y = y[:, 0]  # Convert to mono if stereo

    num_cpu = 1
    slice_size = len(y) // num_cpu

    # Prepare the blocks to be analyzed
    blocks = [
        (y[i * slice_size : (i + 1) * slice_size], i * slice_size, sr)
        for i in range(num_cpu)
    ]
    all_results = []
    with Pool(processes=num_cpu) as pool:
        results = pool.map(process_chunk, blocks)

    comp_time = time.time() - start
    print(f"Computation time: {comp_time}")
    start = time.time()
    # Collect and write the results
    for result in results:
        all_results += result
    df = pd.DataFrame(all_results)
    df.to_csv("statistics.csv", mode="a", header=False, index=False)
    write_time = time.time() - start
    print(f"Write time: {write_time}")


def process_chunk(block):
    y_chunk, start_idx, sr = block
    stats_list = []
    for i in tqdm(range(0, len(y_chunk) - BLOCKSIZE)):
        start = time.time()
        stats = analyze_audio_block((y_chunk[i : i + BLOCKSIZE], start_idx + i, sr))
        print(f"Time: {time.time() - start}")
        if stats is not None and len(stats["major_frequencies"]) > 0:
            stats_list.append(stats)
    return stats_list


if __name__ == "__main__":
    try:
        for i in range(NUM_RUNS):
            stop_mem_recording = False
            audio_file = "/home/kilian/hetcomp/Aufgabe2/nicht_zu_laut_abspielen.wav"

            analyze_audio_blocks(audio_file)
    except KeyboardInterrupt:
        print("Memory recording stopped")
        print("Exiting...")
        exit(0)
