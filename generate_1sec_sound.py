import soundfile
import numpy as np

sample_rate = 44100
soundfile.write("data/1sec.wav", np.zeros(sample_rate),sample_rate)