import numpy as np
from scipy.fftpack import fft
from scipy.signal import medfilt

class OnsetDetection(object):
	"""
	Onset Detection Algorithm that finds the onset times of an audio file. There are three types
	of onset detection algorithms which are : 1) Spectral Flux, 2) High Frequency Content and 3) Complex Domain

	Attributes:
	inputSignal: The audio file
	fs: Sampling rate
	detectionType: The algorithm to calculate the onset detection function
	hopTime: Length of overlapping window in seconds
	fftTime: Length of FFT size in seconds
	threshold: The threshold for pick picking algorithm
	detectionFunc: The output of the onset detection function

	"""

	def __init__(self, inputSignal, fs = 44100, hopTime = 0.01, fftTime = 0.04, threshold = 1.25, detectionType = "Complex Domain", postProcessingType = "Whole"):
		self.inputSignal = inputSignal
		self.fs = fs
		self.hopTime = hopTime
		self.fftTime = fftTime
		self.threshold = threshold
		self.hopSize = round(fs * hopTime)
		self.fftOrder = np.ceil(np.log2(self.fs * self.fftTime))
		self.fftSize = int(pow(2, self.fftOrder))

		self.frameCount = int(np.floor((len(self.inputSignal) - self.fftSize) / self.hopSize))
		self.currentFrameFFT = np.zeros(self.fftSize)
		self.prevFrameFFT = np.zeros(self.fftSize)
		self.prevPrevFrameFFT = np.zeros(self.fftSize)
		self.detectionFunc = np.zeros(self.frameCount)

		self.detectionType = detectionType
		self.postProcessingType = postProcessingType

	@staticmethod
	def highFrequencyContent(currentMag, freqSample, fftSize):
		if (freqSample <= fftSize/2):
			hfc = pow(currentMag, 2) * freqSample
		else:
			hfc = pow(currentMag, 2) * (fftSize - freqSample)
		return hfc

	@staticmethod
	def spectralFlux(currentMag, prevMag):
		if (currentMag > prevMag):
			sf = currentMag - prevMag
		else:
			sf = 0
		return sf


	@staticmethod
	def complexDomain(currentMag, prevMag, currentPhase, targetPhase):
		cd = np.sqrt(pow(prevMag, 2) + pow(currentMag, 2) - 2 * prevMag * currentMag * np.cos(currentPhase - targetPhase))
		return cd


	def processFrame(self):

		# This function outputs the onset detection function of the current frame
		# using self.detectionType algorithm

		result = 0

		for i in range(self.fftSize):
			currentMag = np.abs(self.currentFrameFFT[i]) # get magnitude of current frame's FFT at index i
			prevMag = np.abs(self.prevFrameFFT[i]) # get magnitude of previous frame's FFT at index i
			currentPhase = np.angle(self.currentFrameFFT[i]) # get phase of current frame's FFT at index i
			prevPhase = np.angle(self.prevFrameFFT[i]) # get phase of previous frame's FFT at index i
			prevPrevPhase = np.angle(self.prevPrevFrameFFT[i]) # get phase of the last two frame's FFT at index i
			targetPhase = 2 * prevPhase - prevPrevPhase # calculate target phase

			# Calculate the onset detection function according to the algorithm that the user has set
			# by default is Complex Domain algorithm because it is more robust to different type
			# of instruments and works best for audio with complex mixture compared to other
			# methods

			if (self.detectionType == "Complex Domain"):
				result += self.complexDomain(currentMag, prevMag, currentPhase, targetPhase)
			elif (self.detectionType == "Spectral Flux"):
				result += self.spectralFlux(currentMag, prevMag)
			elif (self.detectionType == "High Frequency Content"):
				result += self.highFrequencyContent(currentMag, i, self.fftSize)

		result /= self.fftSize
		return result			

	def process(self):

		# Process each frame of the audio signal
		for i in range(self.frameCount):
			ptr = i * self.hopSize # points to the first index of current frame
			currentFrame = self.inputSignal[ptr:ptr + self.fftSize] # get current frame
			currentFrame = currentFrame * np.hamming(self.fftSize) # apply hamming window to current frame
			currentFrame = np.append(currentFrame, np.zeros(self.fftSize)) # zero pad the current frame
			self.currentFrameFFT = fft(currentFrame, self.fftSize) # calculate the FFT of the current frame

			self.detectionFunc[i] = self.processFrame() # calculates the onset detection function of this frame

			self.prevPrevFrameFFT = self.prevFrameFFT
			self.prevFrameFFT = self.currentFrameFFT

		return self.detectionFunc

	def postProcessing(self):

		# Do some post processing techniques such as normalising, smoothing and DC removal
		# Later, we could add constraint to avoid "double detection" (e.g. peaks that are close to each other)

		self.process()

		self.detectionFunc /= medfilt(self.detectionFunc, 11) # Normalise with median filtered function

		self.detectionFunc = (self.detectionFunc - np.mean(self.detectionFunc)) / np.std(self.detectionFunc) # Standardise to zero mean

		return self.detectionFunc

