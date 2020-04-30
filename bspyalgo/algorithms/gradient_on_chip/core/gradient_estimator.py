'''Contains function to determine the gradients of the device via the 'lock-in detection method'.
'''

import numpy as np

def gradient_estimator(outputs, sampling_frequency, input_sines, amplitude_sines, phase_threshold = 90):

    t = np.arange(0, outputs.shape[0] / sampling_frequency, 1 / sampling_frequency)
    y_ref1 = np.sin(input_sines[:, np.newaxis] * 2 * np.pi * t)
    y_ref2 = np.sin(input_sines[:, np.newaxis] * 2 * np.pi * t + np.pi / 2)
    
    y_out1 = y_ref1 * (outputs - np.mean(outputs))
    y_out2 = y_ref2 * (outputs - np.mean(outputs))
    
    amp1 = (np.mean(y_out1, axis=1)) 
    amp2 = (np.mean(y_out2, axis=1))
    
    amp_out = 2 * np.sqrt(amp1 ** 2 + amp2 ** 2)
    phase_out = np.arctan2(amp2, amp1) * 180 / np.pi
    sign = 2 * (abs(phase_out) < phase_threshold) - 1
    
    return sign * amp_out / amplitude_sines
