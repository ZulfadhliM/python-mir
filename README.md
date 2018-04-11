# Music Information Retrieval (MIR) in Python

Some state-of-the-art music information retrieval techniques written in Python.

## Features

* Onset Detection: estimates the onset times of an audio signal
* Pitch Estimation: estimates the pitches of a monophonic audio signal

## Getting Started

The main.py file shows an example of using the MIR techniques. The audio signal that is under analysis is the guitar riff from the track "Day Tripper" written by The Beatles in "1" album. The graph below shows the results:

![MIR Result](https://github.com/ZulfadhliM/python-mir/blob/master/screenshots/result.png)

The top graph shows the signal (right channel only) of the opening guitar riff played by George Harrison. The middle graph shows the onset detection function that is calculated using Rectified Complex Domain method (blue line) and the estimated onset times (dotted red lines). The bottom graph shows the estimated fundamental frequencies over time.

## Dependencies

* NumPy
* SciPy
* matplotlib
* Soundfile (optional)

## Future Work

Will add more MIR techniques e.g. polyphonic frequency estimation, spectral peaks extraction, inharmonicity coefficient estimation, music segmentation and etc.

