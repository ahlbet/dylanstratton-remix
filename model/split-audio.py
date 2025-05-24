from pydub import AudioSegment
import os

def split_audio(input_file, output_dir, segment_length_ms=4000):
  audio = AudioSegment.from_wav(input_file)
  audio_length = len(audio)

  os.makedirs(output_dir, exist_ok=True)
  for i in range(0, audio_length, segment_length_ms):
    chunk = audio[i:i + segment_length_ms]
    chunk_filename = os.path.join(output_dir, f"chunk_{i // segment_length_ms:04d}.wav")
    chunk.export(chunk_filename, format="wav")
    print(f"Exported {chunk_filename} ({len(chunk)} ms)")

input_file = "training-samples/25may18.wav"
output_dir = "training-samples/split"
split_audio(input_file, output_dir)    