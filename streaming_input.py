import pyaudio
import wave
import threading
import soundfile as sf
import numpy as np
import torch
from scipy.io.wavfile import write
import sounddevice as sd

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK = 16000
WAVE_OUTPUT_FILENAME = "audio_file_"
file_index = 0

SCRIPTED_MODEL_PATH = './model_scripted.pt'
model = torch.jit.load(SCRIPTED_MODEL_PATH)
model.eval()

device=torch.device('cpu')

stop_ = False
audio = pyaudio.PyAudio()
clear_audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def predict_scripted(data):
  # Use the model to predict the label of the waveform
  numpydata = np.frombuffer(data, dtype=np.float32)
  tensor = torch.tensor(numpydata).unsqueeze(dim=0)
  tensor = tensor.to(device)
  tensor = model(tensor.unsqueeze(0))
  tensor = get_likely_index(tensor)
  return tensor[0].item()

def stop():
    global stop_
    while True:
        if not input('Press Enter >>>'):
            print('exit')
            stop_ = True


t = threading.Thread(target=stop, daemon=True).start()
frames = []

while True:
    data = stream.read(CHUNK)
    silence = bool(predict_scripted(data))

    prev = True
    i = 0 

    if not silence:
        frames.append(data)

        while i < 2:
            data = stream.read(CHUNK)
            silence = bool(predict_scripted(data))

            i = i + 1 if silence and prev else 0
            prev = silence

            frames.append(data)

            if stop_:
                break

        audiodata = np.frombuffer(b''.join(frames), dtype=np.float32)
        write(WAVE_OUTPUT_FILENAME + str(file_index) + '.wav', RATE, audiodata)
        
        print("Saving", WAVE_OUTPUT_FILENAME + str(file_index) + '.wav')
        frames = []
        file_index += 1

    else:
        continue        

    if stop_:
        break

stream.stop_stream()
stream.close()
audio.terminate()

