import soundfile
from matplotlib import pyplot as plt
from OnsetDetection import OnsetDetection
import numpy as np

[inputSignal, fs] = soundfile.read('PianoDebussy.wav')

onsetDetection = OnsetDetection(inputSignal, detectionType = "Complex Domain")
odf = onsetDetection.process()

plt.plot(odf)
plt.show()