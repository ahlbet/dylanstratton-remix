import os
from pydub import AudioSegment


def split_all_wavs(input_dir, output_dir, chunk_length_ms=5000):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            audio = AudioSegment.from_wav(input_path)
            audio_length = len(audio)

            base_name = os.path.splitext(filename)[0]
            file_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(file_output_dir, exist_ok=True)

            print(f"Processing: {filename}")
            for i in range(0, audio_length, chunk_length_ms):
                chunk = audio[i : i + chunk_length_ms]
                chunk_filename = os.path.join(
                    file_output_dir, f"{base_name}_chunk_{i // chunk_length_ms:04d}.wav"
                )
                chunk.export(chunk_filename, format="wav")
                print(f"  Saved: {chunk_filename}")


def main():
    input_file = "data/processed"
    output_dir = "data/processed/split"
    split_all_wavs(input_file, output_dir, 5000)


if __name__ == "__main__":
    main()
