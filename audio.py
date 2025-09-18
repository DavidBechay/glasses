import sounddevice as sd
import wave
import numpy as np
import time
import os

# Settings
CHANNELS = 1
RATE = 44100
CHUNK_DURATION = 60 * 60  # seconds (1 hour)
FORMAT = 'int16'

def record_chunk(filename, duration=CHUNK_DURATION):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(RATE)

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status)
        data = (indata * 32767).astype(np.int16).tobytes()
        wf.writeframes(data)

    with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=callback):
        print(f"Recording {filename} for {duration} seconds...")
        time.sleep(duration)

    wf.close()
    print(f"Saved {filename}")

# Continuous recording loop
file_index = 1
try:
    while True:
        filename = f"recording_{file_index}.wav"
        record_chunk(filename, CHUNK_DURATION)
        file_index += 1
except KeyboardInterrupt:
    print("\nStopped recording.")
