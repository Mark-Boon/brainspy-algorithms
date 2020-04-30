# -*- coding: utf-8 -*-
""" Contains functions that are used to test the accuracy of the on-chip gradient descent method.
Created on Tue 25 Feb 2020
@author: Mark-Boon
"""

import numpy as np
from bspyalgo.algorithms.gradient.core.gradient_estimator import gradient_estimator
from bspyproc.bspyproc import get_processor
from bspyalgo.utils.io import create_directory_timestamp, save

def convergence(inputs, configs):
    '''
        This function is used to measure gradient data with increasing sample time T.
        The uncertainty in gradient information should decrease as 1/T, as shown in the
        equations (integrals that are approximated to be 0 for large enough T decay with a rate of 1/T).

        On all input electrodes a sine wave is added to the DC voltage.

        outputs [T x N X L]:    T amount of sample times
                                N repetitions of experiment (to compute std)
                                L data points
            
        IV_gradients [T x N x C]:   T amount of sample times
                                    N repetitions of experiment
                                    C controls to compute gradient for

        IVmeans [T x C]:        T amount of sample times
                                C controls to compute gradient for
    '''
    # Initialize configs
    waveform = configs["waveform"]
    processor_config = configs["processor"]
    fs = processor_config["sampling_frequency"]

    # Initialize arrays
    sample_times = np.round(np.linspace(configs["sample_times"]["start"], configs["sample_times"]["stop"], configs["sample_times"]["steps"]) / waveform["frequency_factor"], 3)

    t = np.arange(0.0, sample_times[-1], 1/fs)
    outputs = np.zeros((sample_times.shape[0], waveform["repetitions"], int(sample_times[-1] * fs))) # length of outputs is defined by longest measurement
    IV_gradients = np.zeros((sample_times.shape[0], waveform["repetitions"], processor_config["input_electrode_no"]))
    phases = np.zeros((sample_times.shape[0], waveform["repetitions"], processor_config["input_electrode_no"]))

    # Create DC inputs (for the largest sample time, for lower sample times we can take a smaller part of this array)
    input_waveforms = np.array(inputs[:, np.newaxis] * np.ones(int(sample_times[-1] * fs)))

    # Add AC part to inputs
    input_waveforms += np.sin(2 * np.pi * waveform["frequency_factor"] * np.array(waveform["frequencies"])[:, np.newaxis] * t) * np.array(waveform["amplitude_sines"])[:, np.newaxis]

    # Add ramping up and down to the inputs
    ramp = inputs[:, np.newaxis] * np.linspace(0, 1, int(fs * waveform["ramp_length"]))    

    # Data acquisition loop
    print('Starting gradient convergence for increasing sample times experiment. Estimated duration: ' + str(np.sum(waveform["repetitions"] * sample_times) / 60 + waveform["repetitions"] * sample_times.shape[0] * (0.2 + 2 * waveform["ramp_length"]) / 60) + ' minutes.')
    for times in range(sample_times.shape[0]):
        print('Sampling for sample time ' + str(times + 1) + '/' + str(sample_times.shape[0]) + ' (' + str(sample_times[times]) + 's)')

        for repetition in range(waveform["repetitions"]):
            print('Sampling iteration ' + str(repetition + 1) + '/' + str(waveform["repetitions"]) + '...')
            # input waves
            input_signal = np.concatenate((ramp, input_waveforms[:, :int(sample_times[times] * fs)], ramp[:, ::-1]), axis=1).T
            # Get processor
            processor_config["shape"] = input_signal.shape[0]
            processor = get_processor(processor_config)
            ramped_outputs = processor.get_output(input_signal)
            outputs[times, repetition, 0:int(sample_times[times] * fs)] = ramped_outputs[int(fs * waveform["ramp_length"]): -int(fs * waveform["ramp_length"]), 0]
            # Determine gradients using the orthogonality of sine waves:    
            IV_gradients[times, repetition, :] = gradient_estimator(outputs[times, repetition, 0:int(sample_times[times] * fs)], fs, waveform["frequency_factor"] * np.array(waveform["frequencies"]), np.array(waveform["amplitude_sines"]), waveform["phase_threshold"])
    return input_waveforms, outputs, IV_gradients, sample_times

def multiwave_accuracy(inputs, configs):
    '''
        This function is used to determine the accuracy of the gradient information when applying sine waves to all input electrodes simultaneously.
        For an input electrode, increasing frequencies are applied to test how accurate the gradient is at higher frequencies.
        This is done for every single electrode. After this, this is also applied for all electrodes simultaneously.
        This can be used to compare: 1) How fast can we sample. 2) Whether we can determine gradients simulaneously.

        Info on most important arrays:
        input_waveforms:[F x C+1 x C x L]:  
                                            C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                            F different wave factors (to determine the maximum wave frequency which still gives accurate gradients)
                                            C electrodes input dimension
                                            L sample length (differs per factor, if L is lower than the max, it is appended with zeros to fit in the array)
                                        
        data: [F x C+1 x N x L]:            C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                            F different wave factors (to determine the maximum wave frequency while still obtaining accurate data)
                                            N amount of iterations (for some statistics)
                                            L sample length

        IV_gradients: [F x C+1 x N x C]:    For every electrode (and all simultaneously) and for every factor, sample N times and determine gradient w.r.t. each control voltage
    
        Input parameters:
            inputs: 1D array containing a voltage value per input electrode
    '''
    # Initialize configs
    waveform = configs["waveform"]
    processor_config = configs["processor"]
    fs = processor_config["sampling_frequency"]
    # Get processor
    processor = get_processor(processor_config)

    # Initialize arrays
    IV_gradients = np.zeros((len(waveform["frequency_factor"]), processor_config["input_electrode_no"] + 1, waveform["repetitions"], processor_config["input_electrode_no"]))

    input_waveforms = np.zeros((len(waveform["frequency_factor"]), processor_config["input_electrode_no"] + 1, processor_config["input_electrode_no"], int(fs * (2 * waveform["ramp_length"] + waveform["wavelengths"] / (min(waveform["frequency_factor"]) * min(waveform["frequencies"]))))  ))
    outputs = np.zeros((len(waveform["frequency_factor"]), processor_config["input_electrode_no"] + 1, waveform["repetitions"], int(fs * (waveform["wavelengths"] / (min(waveform["frequency_factor"]) * min(waveform["frequencies"])))) ))

    # Create inputs loop
    for factor in range(len(waveform["frequency_factor"])):
        freq = np.array(waveform["frequencies"]) * waveform["frequency_factor"][factor]
        sample_time = waveform["wavelengths"] / min(freq) # Sample time is dependent on the lowest frequency of the current factor and is always a specific amount of periods of the slowest frequency of the input waves.
        t = np.arange(0.0, sample_time, 1 / fs)
        
        pre_inputs = np.ones(processor_config["input_electrode_no"] + 1)[:, np.newaxis, np.newaxis] * (np.array(inputs[:, np.newaxis] * np.ones(int(sample_time * fs)) ))[np.newaxis, :] 
        # Add sine waves on top of DC voltages 
        for g in range(processor_config["input_electrode_no"] + 1):
            indices = g # Only add AC signal to a single electrode
            if g == processor_config["input_electrode_no"]: indices = range(processor_config["input_electrode_no"]) # except for the last measurement, now they all obtain an AC signal
            pre_inputs[g, indices, :] += np.sin(2 * np.pi * freq[indices, np.newaxis] * t[:pre_inputs.shape[2]]) * np.array(waveform["amplitude_sines"])[indices, np.newaxis]
            
        # Add ramping up and down to the inputs for device safety
        ramp = np.ones(processor_config["input_electrode_no"] + 1)[:, np.newaxis, np.newaxis] * (inputs[:, np.newaxis] * np.linspace(0, 1, int(fs * waveform["ramp_length"])))    
        inputs_ramped = np.concatenate((ramp, pre_inputs, ramp[:, :, ::-1]), axis=2)    
        input_waveforms[factor, :, :, 0:int(fs * (2 * waveform["ramp_length"] + sample_time))] = inputs_ramped # Now we have [F, C+1, C, L+ramp_length]

    # Data acquisition loop
    print('Estimated time required for experiment: ' + str(np.sum((processor_config["input_electrode_no"] + 1) * waveform["repetitions"] * waveform["wavelengths"] / (min(waveform["frequencies"]) * np.array(waveform["frequency_factor"]))) / 60 + (processor_config["input_electrode_no"] + 1) * waveform["repetitions"] * len(waveform["frequency_factor"]) * (2 * waveform["ramp_length"] + 0.2) / 60) + ' minutes (total sample time)')
    for factor in range(len(waveform["frequency_factor"])):
        print('Sampling for factor ' + str(waveform["frequency_factor"][factor]))
        freq = np.array(waveform["frequencies"]) * waveform["frequency_factor"][factor]
        sample_time = waveform["wavelengths"] / min(freq) # Sample time is dependent on the lowest frequency of the current factor and is always a specific amount of periods of the slowest frequency of the input waves   
        
        for electrode in range(processor_config["input_electrode_no"] + 1):
            print('Sampling sines for electrode ' + str(electrode + 1) + ' (8 = all electrodes)')
            input_signal = input_waveforms[factor, electrode, :, 0:int(fs * (2 * waveform["ramp_length"] + sample_time))].T
            
            # Get processor. Since for every frequency factor a different input size is used, it is necessary to reinitialize the processor.
            processor_config["shape"] = input_signal.shape[0]
            processor = get_processor(processor_config)

            for iteration in range(waveform["repetitions"]):
                print('Iteration ' + str(iteration + 1) + '/' + str(waveform["repetitions"]) + '...')        
                
                ramped_outputs = processor.get_output(input_signal)
                outputs[factor, electrode, iteration, 0:int(sample_time * fs)] = ramped_outputs[int(fs * waveform["ramp_length"]): -int(fs * waveform["ramp_length"]), 0]
                
                # Determine gradients using the orthogonality of sine waves:              
                IV_gradients[factor, electrode, iteration, :] = gradient_estimator(outputs[factor, electrode, iteration, 0:int(sample_time * fs)], fs, freq, np.array(waveform["amplitude_sines"]), waveform["phase_threshold"])
    return input_waveforms, outputs, IV_gradients


def multiwave_quantitative(configs):
    """
        This function is used to determine the accuracy of the gradient information when applying sine waves to all 
        input electrodes simultaneously for many points in input space.
        
        Info on most important arrays:
        input_waveforms: array([S x C+1 x C x L])]:  
                                            C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                            S different sets of input configurations
                                            C electrodes input dimension
                                            L sample length
                                        
        data:  array([S x C+1 x L])]:   C+1 combinations of wave experiment (C times for the C electrodes, '+1' for applying waves to all electrodes simultaneously)
                                        S different sets of input configurations
                                        L sample length

        IV_gradients: [S x C+1 x C]:    For every electrode (and all simultaneously) and for every CV set, determine gradient w.r.t. each control voltage

    """
    # Initialize configs
    waveform = configs["waveform"]
    processor_config = configs["processor"]
    fs = processor_config["sampling_frequency"]

    t = np.arange(0.0, waveform["wavelengths"] / (waveform["frequency_factor"] * min(waveform["frequencies"])), 1 / fs)

    # Initialize arrays
    IV_gradients = np.zeros((waveform["sets"], processor_config["input_electrode_no"] + 1, processor_config["input_electrode_no"])) 
    input_waveforms = np.zeros((waveform["sets"], processor_config["input_electrode_no"] + 1, processor_config["input_electrode_no"], t.shape[0] + int(2 * fs * waveform["ramp_length"])))
    outputs = np.zeros((waveform["sets"], processor_config["input_electrode_no"] + 1, t.shape[0]))
    processor_config["shape"] = input_waveforms.shape[-1]
    # Create random input voltages
    controls = np.random.random((waveform["sets"], processor_config["input_electrode_no"])) * (np.array(waveform["input_range"])[:, 1] - np.array(waveform["input_range"])[:, 0]) + np.array(waveform["input_range"])[:, 0]

    # Get processor
    processor = get_processor(processor_config)

    # Generate input data
    for input_set in range(waveform["sets"]):
        input_waveforms[input_set, :, :, int(waveform["ramp_length"] * fs): -int(waveform["ramp_length"] * fs)] = controls[input_set, :, np.newaxis] * np.ones((t.shape[0]))[np.newaxis, :] # DC component
        # Add sine waves on top of DC voltages 
        for g in range(processor_config["input_electrode_no"]+1):
            indices = g # Only add the perturbation to a single electrode...
            if g == 7: indices = [0, 1, 2, 3, 4, 5, 6] # ... except for the last measurement, now they are all perturbed.
            input_waveforms[input_set, g, indices, int(waveform["ramp_length"] * fs): -int(waveform["ramp_length"] * fs)] += np.sin(2 * np.pi * waveform["frequency_factor"] * np.array(waveform["frequencies"])[indices, np.newaxis] * t) * np.array(waveform["amplitude_sines"])[indices, np.newaxis]        
        
        # Add ramping up and down to the inputs for device safety
        ramp = np.ones(processor_config["input_electrode_no"] + 1)[:, np.newaxis, np.newaxis] * (controls[input_set, :, np.newaxis] * np.linspace(0, 1, int(fs * waveform["ramp_length"])))    
        input_waveforms[input_set, :, :, 0:int(waveform["ramp_length"] * fs)] = ramp
        input_waveforms[input_set, :, :, -int(waveform["ramp_length"] * fs):] = ramp[:, :, ::-1] 

    # Data acquisition loop
    print('Estimated time required for experiment: ' + str(np.sum((processor_config["input_electrode_no"] + 1) * waveform["sets"] * waveform["wavelengths"] / (waveform["frequency_factor"] * min(waveform["frequencies"]))) / 60 + (processor_config["input_electrode_no"] + 1) * waveform["sets"] * (2 * waveform["ramp_length"] + 0.2) / 60) + ' minutes (total sample time)')
    for input_set in range(waveform["sets"]):
        print('Sampling for set ' + str(input_set + 1))
        
        for g in range(processor_config["input_electrode_no"] + 1):
            print('Sampling sines for electrode ' + str(g + 1) + ' (8 = all electrodes)')    
                
            ramped_outputs = processor.get_output(input_waveforms[input_set, g].T) 
            outputs[input_set, g, :] = ramped_outputs[int(fs * waveform["ramp_length"]): -int(fs * waveform["ramp_length"]), 0]   # Cut off the ramping up and down part.
          
            # Determine gradients using the orthogonality of sine waves           
            IV_gradients[input_set, g, :] = gradient_estimator(outputs[input_set, g, :], fs, waveform["frequency_factor"] * np.array(waveform["frequencies"]), np.array(waveform["amplitude_sines"]), waveform["phase_threshold"])

    return input_waveforms, outputs, IV_gradients, controls, t


def multiwave_accuracy_amplitudes(inputs, configs):
    """
        Varies the magnitude of the sine wave perturbations to determine when the simultaneous gradient estimation becomes inaccurate due to interference.
        Note that it is currently hardcoded for a 7 input - 1 output device. This shouldn't be too hard to generalize.
    """
    # Initialize configs
    waveform = configs["waveform"]
    processor_config = configs["processor"]
    fs = processor_config["sampling_frequency"]

    t = np.arange(0.0, waveform["wavelengths"] / (waveform["frequency_factor"] * min(waveform["frequencies"])), 1 / fs)

    # Initialize arrays
    IV_gradients = np.zeros((len(waveform["amplitude_sines"]), processor_config["input_electrode_no"] + 1, waveform["repetitions"], processor_config["input_electrode_no"])) 
    input_waveforms = np.zeros((len(waveform["amplitude_sines"]), processor_config["input_electrode_no"] + 1, processor_config["input_electrode_no"], t.shape[0] + int(2*fs*waveform["ramp_length"])))
    outputs = np.zeros((len(waveform["amplitude_sines"]), processor_config["input_electrode_no"] + 1, waveform["repetitions"], t.shape[0]))
    processor_config["shape"] = input_waveforms.shape[-1]

    # Get processor
    processor = get_processor(processor_config)

    # Generate input data.
    for input_amplitude in range(len(waveform["amplitude_sines"])):
        input_waveforms[input_amplitude, :, :, int(waveform["ramp_length"] * fs): -int(waveform["ramp_length"] * fs)] = inputs[:, np.newaxis] * np.ones((t.shape[0]))[np.newaxis, :] # DC component.
        # Add sine waves on top of DC voltages. 
        for g in range(processor_config["input_electrode_no"] + 1):
            indices = g # Only add AC signal to a single electrode.
            if g == 7: indices = [0, 1, 2, 3, 4, 5, 6] # Except for the last measurement, now they all obtain an AC signal.
            input_waveforms[input_amplitude, g, indices, int(waveform["ramp_length"] * fs): -int(waveform["ramp_length"] * fs)] += np.sin(2 * np.pi * waveform["frequency_factor"] * np.array(waveform["frequencies"])[indices, np.newaxis] * t) * np.array(waveform["amplitude_sines"])[input_amplitude][indices, np.newaxis] 
        
        # Add ramping up and down to the inputs for device safety.
        ramp = np.ones(processor_config["input_electrode_no"] + 1)[:, np.newaxis, np.newaxis] * (inputs[:, np.newaxis] * np.linspace(0, 1, int(fs * waveform["ramp_length"])))    
        input_waveforms[input_amplitude, :, :, 0: int(waveform["ramp_length"] * fs)] = ramp
        input_waveforms[input_amplitude, :, :, -int(waveform["ramp_length"] * fs):] = ramp[:, :, ::-1] 

    # Data acquisition loop.
    for input_set in range(len(waveform["amplitude_sines"])):
        print('Sampling for set ' + str(input_set + 1))
        
        for g in range(processor_config["input_electrode_no"] + 1):
            print('Sampling sines for electrode ' + str(g + 1) + ' (8 = all electrodes)') 
            for it in range(waveform["repetitions"]):
                print("Iteration " + str(it+1) + "/" + str(waveform["repetitions"]))
               
                ramped_outputs = processor.get_output(input_waveforms[input_set, g].T) 
                outputs[input_set, g, it, :] = ramped_outputs[int(fs * waveform["ramp_length"]): -int(fs * waveform["ramp_length"]), 0]   # Cut off the ramping up and down part
            
                # Determine gradients              
                IV_gradients[input_set, g, it, :] = gradient_estimator(outputs[input_set, g, it, :], fs, waveform["frequency_factor"] * np.array(waveform["frequencies"]), np.array(waveform["amplitude_sines"])[input_set], waveform["phase_threshold"])

    return input_waveforms, outputs, IV_gradients, t


def driver_sync_test(configs):
    import matplotlib.pyplot as plt

    waveform = configs["waveform"]
    processor_config = configs["processor"]
    fs = processor_config["sampling_frequency"]
    t = np.arange(0.0, waveform["wavelengths"] / (np.array(waveform["frequency_factor"]) * min(waveform["frequencies"])), 1 / fs)
    inputs = np.array(waveform["amplitude_sines"])[:, np.newaxis] * np.sin(2 * np.pi * np.array(waveform["frequencies"])[:, np.newaxis] * waveform["frequency_factor"] * t)

    processor_config["shape"] = inputs.shape[1]
    processor = get_processor(processor_config)

    outputs = processor.get_output(inputs.T)

    for i in range(len(waveform["frequencies"])):
        plt.figure()
        plt.plot(inputs[i])
        plt.plot(outputs)
    plt.show()
