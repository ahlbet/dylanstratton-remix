#!/usr/bin/env python3
"""
Template Audio Preprocessing Script

- Loads WAVs from INPUT_DIR
- Resamples to SAMPLE_RATE and converts to mono
- Trims silence, applies band-pass filtering
- Normalizes to [-1,1]
- Pads or trims to fixed length
- Creates N_AUG augmented versions (time-stretch, pitch-shift, noise)
- Saves to OUTPUT_DIR
"""

import os
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, sosfiltfilt

# ——— Config ———
INPUT_DIR = Path("data_raw")
OUTPUT_DIR = Path("data_processed")
SAMPLE_RATE = 16000
TARGET_DURATION = 30.0  # seconds per clip
TARGET_LEN = int(SAMPLE_RATE * TARGET_DURATION)
N_AUG = 1  # augmentations per source file

# Augmentation parameters
TIME_STRETCH_RANGE = (0.8, 1.2)  # +/-10%
PITCH_SHIFT_STEPS = (-12, 12)  # semitones
NOISE_LEVEL = 0.01  # relative amplitude


# ——— Helper Functions ———
def butter_filter(lowcut, highcut, fs, order=6, btype="band"):
    nyq = 0.5 * fs
    if btype == "band":
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
    else:
        cutoff = (lowcut or highcut) / nyq
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=btype)
    return b, a


def apply_filter(audio, fs):
    """
    Apply a high-pass filter at 20 Hz to remove DC/rumble.
    If you really want band-pass, uncomment the bandpass block.
    """
    # 1) High-pass at 20 Hz
    sos_hp = butter(6, 20, btype="highpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos_hp, audio)

    # 2) (Optional) band-pass 20 Hz–7.5 kHz — stays below Nyquist
    # sos_bp = butter(6, [20, 7500], btype='bandpass', fs=fs, output='sos')
    # filtered = sosfiltfilt(sos_bp, filtered)

    return filtered


def trim_silence(audio):
    # trimmed, _ = librosa.effects.trim(audio, top_db=60)
    return audio


def normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-9)


def pad_or_trim(audio, target_len):
    if len(audio) < target_len:
        pad_width = target_len - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")
    else:
        audio = audio[:target_len]
    return audio


def process_file(path):
    audio, sr = librosa.load(path, sr=None, mono=True)
    print(
        f"\n[{path.name}] raw load → {len(audio)} samples @ {sr} Hz; max amp = {np.max(np.abs(audio)):.6f}"
    )
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    orig = audio.copy()
    # Trim silence
    trimmed, _ = librosa.effects.trim(audio, top_db=20)
    audio = trimmed if len(trimmed) > 0 else audio

    # Filter
    try:
        filtered = apply_filter(audio, SAMPLE_RATE)
        # guard against zeros or non-finite
        if (not np.isfinite(filtered).all()) or np.max(np.abs(filtered)) < 1e-6:
            audio = orig
        else:
            audio = filtered
    except Exception:
        # any error in filtering → use original
        audio = orig
    # Replace any NaN/Inf with zeros
    audio = np.nan_to_num(audio, posinf=0.0, neginf=0.0)

    # Normalize & pad/trim
    audio = normalize(audio)
    audio = pad_or_trim(audio, TARGET_LEN)

    return audio


def augment(audio, sr):
    # Make sure buffer is clean
    audio = np.nan_to_num(audio, posinf=0.0, neginf=0.0)
    audio = audio.astype(np.float32)

    variants = []
    # 1) time-stretch (keyword-only now)
    rate = np.random.uniform(*TIME_STRETCH_RANGE)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    variants.append(pad_or_trim(stretched, TARGET_LEN))

    # 2) pitch-shift
    steps = np.random.uniform(*PITCH_SHIFT_STEPS)
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
    variants.append(pad_or_trim(shifted, TARGET_LEN))

    # 3) add Gaussian noise
    noise = np.random.randn(len(audio)) * NOISE_LEVEL
    noisy = audio + noise
    variants.append(pad_or_trim(noisy, TARGET_LEN))

    return variants


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for wav_path in INPUT_DIR.glob("*.wav"):
        base = wav_path.stem
        audio = process_file(wav_path)

        # save cleaned clip
        clean_out = OUTPUT_DIR / f"{base}_clean.wav"
        sf.write(clean_out, audio, SAMPLE_RATE)

        # save augmentations
        augs = augment(audio, SAMPLE_RATE)
        for i, aug_audio in enumerate(augs[:N_AUG], start=1):
            aug_out = OUTPUT_DIR / f"{base}_aug{i}.wav"
            sf.write(aug_out, aug_audio, SAMPLE_RATE)

        print(f"Processed {wav_path.name}: cleaned + {N_AUG} augmentations saved.")


if __name__ == "__main__":
    main()
