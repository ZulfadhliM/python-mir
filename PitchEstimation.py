import numpy as np
from matplotlib import pyplot as plt

class PitchEstimation(object):
	"""
	Pitch Estimation Algorithm that finds the pitches (in Hz) of an audio file. So far, there is only one type
	of pitch estimation algorithm, which is the YIN method that works well with monophonic signals.

	Attributes:
	inputSignal: The audio file
	fs: Sampling rate (default = 44100 Hz)
	windowSize: Length of window size in seconds (default = 80ms)
	hopTime: Length of overlapping window in seconds (default = 10ms)
	threshold: The threshold for voice or unvoiced decision (default = 0.2)
	minFreq: The minimum frequency for the search range (default = 65 Hz, low C note)
	timeStamp: An array of the time stamps for each frame

	"""
	def __init__(self, inputSignal, fs = 44100, windowSize = 0.08, hopTime = 0.01, minFreq = 65, threshold = 0.2):

		self.fs = fs
		self.maxLag = int(np.ceil(fs / minFreq) + 1)
		self.threshold = threshold
		self.windowSize = round(fs * windowSize)
		self.hopSize = round(fs * hopTime)
		self.frameCount = int(np.floor((len(inputSignal) - self.windowSize) / self.hopSize))		
		self.inputSignal = np.append(inputSignal, np.zeros(self.windowSize - int(len(inputSignal)/self.hopSize)))
		self.timeStamp = np.linspace(0, (self.frameCount - 1) * self.hopSize / self.fs, self.frameCount)
		self.ptr = 0

	def getYIN(self):

		d = np.zeros(self.maxLag - 1)

		for j in range(len(d)):
			d[j] = np.sum(pow(self.inputSignal[self.ptr: self.ptr + self.windowSize] - self.inputSignal[self.ptr + j + 1: self.ptr + self.windowSize + j + 1], 2))
		
		yin = d / np.cumsum(d) * np.arange(1, self.maxLag)

		return yin

	def getPitchOfFrame(self):

		yin = self.getYIN()

		idxBelowThresh = np.where(yin < self.threshold)[0]
		pitch = 0

		if (len(idxBelowThresh) != 0):
			stopAt = np.where(np.diff(idxBelowThresh) > 1)[0]
			
			if (len(stopAt) == 0):
				idxMin = np.argmin(yin[idxBelowThresh])
			else:
				searchRange = idxBelowThresh[0:int(stopAt)-1]
				idxMin = np.argmin(yin[searchRange])

			idx = idxBelowThresh[idxMin]

			num = yin[idx - 1] - yin[idx + 1]
			den = yin[idx - 1] - 2 * yin[idx] + yin[idx + 1]

			if (den != 0):
				pitch = self.fs / (idx + num / den / 2)

		return pitch

	def process(self):
		# Process each frame of the audio signal
		pitches = np.zeros(self.frameCount)
		for i in range(self.frameCount):
			self.ptr = i * self.hopSize # points to the first index of current frame
			pitches[i] = self.getPitchOfFrame()

		return pitches

	def getPitches(self):

		pitches = self.process()

		return pitches