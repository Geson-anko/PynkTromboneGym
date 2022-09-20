import numpy as np
import soundfile

sample_rate = 44100
soundfile.write("data/1sec.wav", np.zeros(sample_rate), sample_rate)
