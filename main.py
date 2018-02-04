import soundfile
from matplotlib import pyplot as plt
from OnsetDetection import OnsetDetection
from PeakPicking import PeakPicking
import numpy as np

[inputSignal, fs] = soundfile.read('PianoDebussy.wav')

onsetDetection = OnsetDetection(inputSignal, detectionType = "Spectral Flux")
odf = onsetDetection.postProcessing()

plt.plot(odf)
plt.show()
