import numpy as np
from bspyalgo.utils.io import create_directory_timestamp, save

class GDData:
    def __init__(self, inputs, targets, nr_epochs, processor, validation_data=(None, None), mask=None):
        assert inputs.shape[1] == len(targets), f'No. of input data {inputs.shape[1]} does not match no. of targets {len(targets)}'
        self.results = {}
        self.results['inputs'] = inputs
        self.results['targets'] = targets
        self.nr_epochs = nr_epochs
        self.results['processor'] = processor
        if mask is None or len(mask) <= 1:
            mask = np.ones(targets.shape[0], dtype=bool)
        self.results['mask'] = mask
        if validation_data[0] is not None and validation_data[1] is not None:
            assert len(validation_data[0]) == len(validation_data[1]), f'No. of validation input data {len(validation_data[0])} does not match no. of validation targets {len(validation_data[1])}'
            self.results['inputs_val'] = validation_data[0]
            self.results['targets_val'] = validation_data[1]
            self.results['performance_history'] = np.zeros((self.nr_epochs, 2))
            print('VALIDATION DATA IS AVAILABLE')
        else:
            self.results['performance_history'] = np.zeros((self.nr_epochs, 1))

    def update(self, current_state): 
        it = current_state['iteration']
        if 'input_waveforms' in current_state.keys():
            self.results['input_waveforms'][it] = current_state['input_waveforms']
        if 'ramp_mask' in current_state.keys():
            self.results['ramp_mask'] = current_state['ramp_mask']
        if 'controls' in current_state.keys():
            if it+1 < self.results['controls'].shape[0]:
                self.results['controls'][it+1] = current_state['controls']
            else:
                print("Last iteration, controls no longer updated.")
        if 'outputs' in current_state.keys():
            self.results['outputs'][it] = current_state['outputs']
        if 'best_outputs' in current_state.keys():
            self.results['best_outputs'] = current_state['best_outputs']
        if 'best_error' in current_state.keys():
            self.results['best_error'] = current_state['best_error']
        if 'best_inputs' in current_state.keys():
            self.results['best_inputs'] = current_state['best_inputs']
        if 'error' in current_state.keys():
            self.results['error'][it] = current_state['error']
        if 'IVgradients' in current_state.keys():
            self.results['IVgradients'][it] = current_state['IVgradients']
        if 'EIgradients' in current_state.keys():
            self.results['EIgradients'][it] = current_state['EIgradients']
        if 'EVgradients' in current_state.keys():
            self.results['EVgradients'][it] = current_state['EVgradients']
        if 'sign' in current_state.keys():
            self.results['sign'][it] = current_state['sign']

    def init_arrays(self, configs):
        ''' Creates placeholders for on-chip GD experiment '''
        hyperparams = configs["hyperparameters"]
        waveform = configs["waveform"]
        processor_config = configs["processor"]
        nr_control_voltages = processor_config["input_electrode_no"] - len(processor_config["input_indices"])
        self.results['input_waveforms'] = np.zeros((hyperparams['nr_epochs'], processor_config["input_electrode_no"], self.results['inputs'].shape[1] + 2 * int(processor_config['sampling_frequency'] * waveform['ramp_length'])))
        self.results['outputs'] = np.zeros((hyperparams['nr_epochs'], self.results['input_waveforms'].shape[2]))
        self.results['sign'] = np.zeros((hyperparams['nr_epochs'], waveform['input_cases'], nr_control_voltages))
        self.results['IVgradients'] = np.zeros((hyperparams['nr_epochs'], waveform['input_cases'], nr_control_voltages))
        self.results['EIgradients'] = np.zeros((hyperparams['nr_epochs'], len(self.results['targets'][self.results["mask"]])))
        self.results['EVgradients'] = np.zeros((hyperparams['nr_epochs'], nr_control_voltages))
        self.results['controls'] = np.zeros((hyperparams['nr_epochs'], nr_control_voltages))
        self.results['error'] = np.ones(hyperparams['nr_epochs'])
        # first set of control voltages randomly initialized within the CV range
        self.results['controls'][0] = np.random.random(nr_control_voltages) * (np.array(processor_config['CVrange'])[:, 1] - np.array(processor_config['CVrange'])[:, 0]) + np.array(processor_config['CVrange'])[:, 0]

    def save_data(self, configs, path, data_type='pickle'):
        if data_type == 'pickle':
            name = 'data.pickle'
        elif data_type == 'numpy':
            name = 'data.npz'

        save(data_type, path, name, timestamp = False,
                                data = {'configs': configs,
                                    'input_waveforms': self.results['input_waveforms'],
                                    'targets': self.results['targets'],
                                    'outputs': self.results['outputs'],
                                    'ramp_mask': self.results['ramp_mask'],
                                    'inputs': self.results['inputs'],
                                    'IVgradients': self.results['IVgradients'],
                                    'EIgradients': self.results['EIgradients'],
                                    'EVgradients': self.results['EVgradients'],
                                    'sign': self.results['sign'],
                                    'controls': self.results['controls'],
                                    'error': self.results['error'],
                                    'path': path,  
                                    'best_inputs': self.results['best_inputs'],
                                    'best_error': self.results['best_error'],
                                    'best_outputs': self.results['best_outputs']                                                                   
                                    })        
        return path