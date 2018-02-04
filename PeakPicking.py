
class PeakPicking(object):

	"""
	Peak picking algorithm: simple algorithm that finds the locations and amplitudes of the peaks in the data
	
	"""

	def __init__(self, data, threshold = 0):
		self.data = data
		self.threshold = threshold
		self.peakLocations = []
		self.peakAmplitudes = []

	def getPeaks(self):

		for i in range(1, len(self.data) - 1):

			if (self.data[i] > self.data[i-1]) and (self.data[i] > self.data[i+1]) and (self.data[i] > self.threshold):
				self.peakLocations.append(i)
				self.peakAmplitudes.append(self.data[i])

		return self.peakLocations, self.peakAmplitudes


