import soundfile
from matplotlib import pyplot as plt
from OnsetDetection import OnsetDetection
from PeakPicking import PeakPicking
import numpy as np

[inputSignal, fs] = soundfile.read('PianoDebussy.wav')

onsetDetection = OnsetDetection(inputSignal, detectionType = "Complex Domain")
onsetTimes = onsetDetection.getOnsetTimes()

print(onsetTimes)
