import numpy as np

def GenWaveform(amplitudes, lengths, slopes=None):
    '''Generates a waveform with constant intervals of value amplitudes[i] for interval i of length[i]. The slopes argument
    is the number of points of the slope.
    '''
    wave = []
    if len(amplitudes)==len(lengths):  
        for i in range(len(amplitudes)):
            wave += [amplitudes[i]]*lengths[i]
            if (slopes is not None) and (i < (len(amplitudes)-1)):
                try:
                    wave += np.linspace(amplitudes[i],amplitudes[i+1],slopes[i]).tolist()
                except:
                    wave += np.linspace(amplitudes[i],amplitudes[i+1],slopes[0]).tolist()
            
    elif len(lengths)==1:
        for i in range(len(amplitudes)):
            wave += [amplitudes[i]]*lengths[0]
            if (slopes is not None) and (i < (len(amplitudes)-1)):
                assert len(slopes) == 1, 'slopes argument must have length 1 since len(lengths)=1'
                wave += np.linspace(amplitudes[i],amplitudes[i+1],slopes[0]).tolist()
    else:
        assert 0==1, 'Assignment of amplitudes and lengths is not unique!'

    return wave

def featureExtractor(feature, signallength, edgelength=0.01, fs=1000):
    '''
    Creates inputs and targets for the feature extractor task.
    16 features are made: 0000, 0001, ..., 1111.
    
    inputs:
        feature:        which of the 16 features is being learned (number 0 to 15).
        signallength:   length of total signal.
        edgelength:     
    
    outputs:
        t:  time signal
        x:  [4 x many] inputs
        W:  [many] weights (1 for inputs, 0 for edges)
        
    '''
    assert feature >= 0, "Feature number must be 0 or higher"
    assert feature < 16, "Feature number must be 15 or lower"
    
    signallength = signallength/16 
    
    samples = 16 * round(fs * signallength) + 15 * round(fs * edgelength)
    t = np.linspace(0, samples/fs, samples)
    x = np.zeros((4, samples))
    W = np.ones(samples,dtype=bool)
    target = np.zeros(samples)
    edges = np.zeros(samples,dtype=bool)
    
    x[0] = np.asarray(GenWaveform([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[1] = np.asarray(GenWaveform([0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[2] = np.asarray(GenWaveform([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[3] = np.asarray(GenWaveform([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],[round(fs * signallength)],[round(fs * edgelength)]))
    
    for i in range(1,16):
        edges[round(fs*(i*signallength + (i-1)*edgelength)):round(fs*((i)*signallength + i*edgelength))] = True  
    W[edges] = 0 
    target[round(fs*(feature*signallength + max(0,(feature-1))*edgelength)):round(fs*((feature+1)*signallength + (feature)*edgelength))] = 1
    return t, x, W, target

def featureExtractorAlt(feature, signallength, edgelength=0.01, fs=1000):
    '''
    Alternative encoding for the 2x2 feature extractor.
    2 bit info is stored in a single input.
    For example, create mapping -1 + f1 + f2 such that both input 1 and input 2 contain:
        f1 f2 (1, 0.5)
        0,  0 --->  -1V
        0,  1 --->  -0.5V
        1,  0 --->   0V
        1,  1 --->   0.5V
       
    inputs:
        feature: which of the 16 features is being learned (number 0 to 15).
        signallength: length of total signal.
        edgelength: length between cases (in s).
    
    outputs:
        t:  time signal.
        x:  [2 x many] inputs.
        W:  [many] weights (1 for inputs, 0 for edges).
        
    '''
    assert feature >= 0, "Feature number must be 0 or higher"
    assert feature < 16, "Feature number must be 15 or lower"
    
    signallength = signallength/16 # since Boolean logic works with total signal length, we divide by 16 for a single feature input
    
    samples = 16 * round(fs * signallength) + 15 * round(fs * edgelength)
    t = np.linspace(0, samples/fs, samples)
    x = np.zeros((2, samples))
    W = np.ones(samples,dtype=bool)
    target = np.zeros(samples)
    edges = np.zeros(samples,dtype=bool)
    
    x[0] = np.asarray(GenWaveform([-1,-1,-1,-1,-.5,-.5,-.5,-.5,0,0,0,0,.5,.5,.5,.5],[round(fs * signallength)],[round(fs * edgelength)]))
    x[1] = np.asarray(GenWaveform([-1,-.5,0,.5,-1,-.5,0,.5,-1,-.5,0,.5,-1,-.5,0,.5],[round(fs * signallength)],[round(fs * edgelength)]))

    for i in range(1,16):
        edges[int(fs*(i*signallength + (i-1)*edgelength)):int(fs*((i)*signallength + i*edgelength))] = True  
    W[edges] = 0 
    target[round(fs*(feature*signallength + max(0,(feature-1))*edgelength)):round(fs*((feature+1)*signallength + (feature)*edgelength))] = 1
    return t, x, W, target

def booleanLogic(gate, signallength, edgelength=0.01, fs=1000):
    '''
    inputs:
        gate:           string containing 'AND', 'OR', ...
        signallength:   length of total signal.
        edgelength:     ramp time (in s) between input cases.
    
    '''    
    signallength = signallength/4
    samples = 4 * round(fs * signallength) + 3 * round(fs * edgelength)
    t = np.linspace(0, samples/fs, samples)
    x = np.zeros((2, samples))
    W = np.ones(samples,dtype=bool)
    target = np.zeros(samples)
    edges = np.zeros(samples,dtype=bool)
    
    x[0] = np.asarray(GenWaveform([0,0,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    x[1] = np.asarray(GenWaveform([0,1,0,1],[round(fs * signallength)],[round(fs * edgelength)]))
    for i in range(1,4):
        edges[int(fs*(i*signallength + (i-1)*edgelength)):int(fs*((i)*signallength + i*edgelength))] = True
    W[edges] = 0 
    if gate == 'AND':
        target = np.asarray(GenWaveform([0,0,0,1],[round(fs * signallength)],[round(fs * edgelength)]))
    elif gate == 'OR':
        target = np.asarray(GenWaveform([0,1,1,1],[round(fs * signallength)],[round(fs * edgelength)]))
    elif gate == 'NAND':
        target = np.asarray(GenWaveform([1,1,1,0],[round(fs * signallength)],[round(fs * edgelength)]))
    elif gate == 'NOR':
        target = np.asarray(GenWaveform([1,0,0,0],[round(fs * signallength)],[round(fs * edgelength)]))
    elif gate == 'XOR':
        target = np.asarray(GenWaveform([0,1,1,0],[round(fs * signallength)],[round(fs * edgelength)]))
    elif gate == 'XNOR':
        target = np.asarray(GenWaveform([1,0,0,1],[round(fs * signallength)],[round(fs * edgelength)]))
    else:
        assert False, "Target gate not specified correctly."
    return t, x, W, target