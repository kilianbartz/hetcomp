use hound;
use itertools_num::linspace;
use rustfft::{num_complex::Complex, FftPlanner};
use std::io::Write;
use tqdm::tqdm;

/// Minimal example.
fn main() {
    const BLOCK_SIZE: usize = 2048;
    const BLOCK_SIZE_F: f32 = BLOCK_SIZE as f32;

    // Get the path of the WAV file from the command line arguments
    let wav_path = "/home/kilian/hetcomp/Aufgabe2/nicht_zu_laut_abspielen.wav";

    // Open the WAV file
    let mut reader = hound::WavReader::open(wav_path).expect("Failed to open WAV file");

    // Create a vector to store the samples
    let samples = reader.samples::<i16>();
    let mut samples_vec: Vec<Complex<f32>> = samples
        .into_iter()
        .map(|sample| Complex {
            re: sample.unwrap() as f32,
            im: 0.0,
        })
        .collect();
    if reader.spec().channels == 2 {
        // If the WAV file has two channels, we only use the first channel
        samples_vec = samples_vec
            .into_iter()
            .step_by(2)
            .collect::<Vec<Complex<f32>>>();
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(BLOCK_SIZE);
    let frequencies: Vec<f32> =
        linspace(0.0, reader.spec().sample_rate as f32 / 2.0, BLOCK_SIZE / 2).collect();
    let mut stats = Vec::new();
    for i in tqdm(0..samples_vec.len() - BLOCK_SIZE) {
        let buffer = &mut samples_vec[i..i + BLOCK_SIZE].to_vec();
        fft.process(buffer);
        let mut block_stats = Vec::new();
        for j in 0..BLOCK_SIZE / 2 {
            let freq = frequencies[j];
            let real = buffer[j].re;
            let imag = buffer[j].im;
            let magnitude = 2. / BLOCK_SIZE_F * (real * real + imag * imag).sqrt();
            let magnitude_db = 20.0 * magnitude.log10();
            if magnitude_db > 50. {
                block_stats.push((freq as u32, magnitude_db as u32));
            }
        }
        stats.push((i as u64, block_stats));
    }
    // write stats to text file. Format should be: one line per block, startindex of block, all major frequencies
    let mut file = std::fs::File::create("stats.txt").expect("Failed to create file");
    for block in stats.iter() {
        let mut line = format!("{}\t", block.0);
        for (freq, mag) in block.1.iter() {
            line.push_str(&format!("{}:{},", freq, mag));
        }
        line.push_str("\n");
        file.write_all(line.as_bytes())
            .expect("Failed to write to file");
    }
    file.flush().expect("Failed to flush file");
}
