import soundfile
from matplotlib import pyplot as plt
from OnsetDetection import OnsetDetection

[inputSignal, fs] = soundfile.read('PianoDebussy.wav')

onsetDetection = OnsetDetection(inputSignal)
odf = onsetDetection.process()

plt.plot(odf)
plt.show()