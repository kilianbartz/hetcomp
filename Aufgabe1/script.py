import numpy as np
import matplotlib.pyplot as plt
import librosa

def plot_fourier_transform(audio_file, start_position, block_size):
    # Laden der Audiodatei
    y, sr = librosa.load(audio_file, sr=None)

    # Konvertieren der Startposition von Sekunden zu Samples
    start_sample = int(start_position * sr)
    
    # Extrahieren des gewünschten Blocks
    y_block = y[start_sample:start_sample + block_size]

    # Berechnung der Fourier-Transformierten
    Y = np.fft.fft(y_block)
    Y_magnitude = np.abs(Y)[:block_size // 2]
    frequencies = np.fft.fftfreq(block_size, 1 / sr)[:block_size // 2]

    # Plotten der Ergebnisse
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, Y_magnitude, 'b')
    plt.xlabel('Frequenz (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Fourier-Transformation')
    plt.grid()
    plt.show()

# Parameter für die Analyse
audio_file = 'Geheimnisvolle_Wellenlaengen.wav'  # Pfad zur Audiodatei
start_position = 40.0  # Startposition in Sekunden
block_size = 2048  # Blockgröße in Samples

plot_fourier_transform(audio_file, start_position, block_size)
