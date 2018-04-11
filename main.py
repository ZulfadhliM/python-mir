import soundfile
from matplotlib import pyplot as plt
from OnsetDetection import OnsetDetection
from PitchEstimation import PitchEstimation
import numpy as np

[inputSignal, fs] = soundfile.read('The Beatles - Day Tripper.wav')
inputSignal = inputSignal[0: int(3.7 * fs), 1]

t = np.linspace(0, len(inputSignal) / fs, len(inputSignal))

onsetDetection = OnsetDetection(inputSignal)
onsetTimes = onsetDetection.getOnsetTimes()

pitchEstimation = PitchEstimation(inputSignal)
pitches = pitchEstimation.getPitches()

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(t, inputSignal)
plt.ylabel('Level')

plt.subplot(3, 1, 2)
plt.plot(onsetDetection.timeStamp, onsetDetection.detectionFunc)
plt.ylabel('Onset Detection Function')
for i in range(len(onsetTimes)):
	plt.plot([onsetTimes[i], onsetTimes[i]], [-1, 6.2], '--r')
plt.ylim(-1, 6.2)
plt.subplot(3, 1, 3)
for i in range(len(pitches) - 1):
	plt.plot(pitchEstimation.timeStamp[i: i + 2], [pitches[i], pitches[i]], linewidth = 3, color='blue')
plt.ylim(65, max(pitches) + 10)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')

plt.show()
