import numpy as np
import os
from tqdm import trange
from bspyproc.bspyproc import get_processor
from bspyalgo.utils.io import create_directory_timestamp, save
from bspyalgo.algorithms.gradient_on_chip.core.data import GDData
from bspyalgo.algorithms.gradient_on_chip.core.losses_numpy import choose_loss_function, choose_grad_function 
from bspyalgo.algorithms.gradient_on_chip.core.optim_numpy import get_optimizer 
from bspyproc.utils.control import merge_inputs_and_control_voltages_in_numpy, get_control_voltage_indices
from bspyalgo.algorithms.gradient_on_chip.core.gradient_estimator import gradient_estimator

class OnChipGD():
    def __init__(self, configs):
        self.configs = configs       
        self.load_configs(configs)
        self.init_processor(configs)
        self.init_optimizer()
        
    def load_configs(self, configs):
        self.hyperparams = configs["hyperparameters"]
        self.waveform = configs["waveform"]
        self.processor_config = configs["processor"]

    def init_optimizer(self):
        self.loss_fn = choose_loss_function(self.hyperparams['loss_function'])
        self.grad_fn = choose_grad_function(self.hyperparams['loss_function'])
        self.optimizer = get_optimizer(self.hyperparams) 

    def init_processor(self, configs):
        self.nr_input_voltages = self.processor_config['input_electrode_no']
        self.input_indices = self.processor_config['input_indices']
        self.nr_control_voltages = self.nr_input_voltages - len(self.input_indices)
        self.control_voltage_indices = get_control_voltage_indices(self.input_indices, self.nr_input_voltages)
        self.processor = get_processor(configs["processor"])
        self.clipvalue = configs['processor']['output_clipping_value'] * self.processor.get_amplification_value()

    def optimize(self, inputs, targets, validation_data=(None, None), data_info=None, mask=None):
        '''
            inputs = The inputs of the algorithm. They need to be in numpy. The algorithm also requires the input to be a waveform.
            targets = The targets to which the algorithm will try to fit the inputs. They need to be in numpy.
            validation_data = In some cases, it is required to provide the validation data in the form of (training_data, validation_data)
            mask = In cases where the input is a waveform, the mask helps filtering the slopes of the waveform
        '''
        x_train = inputs * self.waveform['input_scale'] + self.waveform['input_offset']
        y_train = targets * (self.hyperparams['target_high'] - self.hyperparams['target_low']) + self.hyperparams['target_low']
        
        data = GDData(x_train, y_train, self.hyperparams['nr_epochs'], self.processor, validation_data, mask=mask)
        data.init_arrays(self.configs)

        looper = trange(self.hyperparams['nr_epochs'], desc='Initialising')
        last = False
        for epoch in looper:
            input_waveforms, ramp_mask = self.generate_input_waves(x_train, data.results['controls'][epoch], last)
            outputs = self.processor.get_output(input_waveforms.T) [:, 0]
            error = self.loss_fn(outputs[ramp_mask][mask], y_train[mask]) 
            data = self.update_controls(epoch, outputs[ramp_mask][mask], y_train[mask], data)

            data.update({'iteration': epoch, 'input_waveforms': input_waveforms, 'ramp_mask': ramp_mask, 'outputs': outputs, 'error': error})

            description = ' Epoch: ' + str(epoch) + ' Error:' + str(error)
            looper.set_description(description)
            if error < self.hyperparams['stop_threshold']:
                print("\n stopping criterium met, stopping experiment...")
                break
            #TODO: Make option to create a figure of progress every few iterations

        if 'experiment_name' not in self.configs:
            self.configs['experiment_name'] = 'experiment'
        if 'results_path' in self.configs:
            self.dir_path = create_directory_timestamp(self.configs['results_path'], self.configs['experiment_name'] + str(self.configs['task']))
        else:
            self.dir_path = create_directory_timestamp(os.path.join('tmp', 'dump'), self.configs['experiment_name'])

        self.measure_best(x_train, y_train, data)
        data.path = data.save_data(self.configs, self.dir_path)
        return data

    def generate_input_waves(self, inputs, controls, last=False): #TODO: rewrite to use already written code from bspy processors
        ''' 
            Generates the waveforms of the input (task inputs + controls) data, including the sine waves used to determine the gradients
            and the ramping between different input cases as well as the ramping up and down before and after the measurement.
            inputs: input waveforms (including ramping between input cases)
        '''
        fs = self.processor_config['sampling_frequency']
        freq = np.array(self.waveform['frequencies']) * self.waveform["frequency_factor"]
        ramp_time = self.waveform['ramp_length'] # in s 
        slope_time = self.waveform['slope_lengths'] # in s 
        amplitude_lengths = self.waveform['amplitude_lengths']
        amplitude_sines = np.array(self.waveform['amplitude_sines'])
        input_waveforms = np.zeros((self.nr_control_voltages + len(self.input_indices), inputs.shape[1] + 2 * int(fs * ramp_time)))
        ramp_mask = np.zeros(inputs.shape[1] + 2 * int(fs * ramp_time), dtype=bool)
        ramp_mask[int(fs * ramp_time):-int(fs * ramp_time)] = 1

        # Apply DC control voltages:
        input_waveforms[self.control_voltage_indices, int(fs * ramp_time):-int(fs * ramp_time)] = controls[:, np.newaxis] * np.ones(inputs.shape[1])

        # Add sine waves to cont voltages for each input case
        for case in range(self.waveform['input_cases']): # The sine waves for each input case start from phase = 0. During ramping there is no sine wave.
            start = int(fs * ramp_time + case * fs * (slope_time + amplitude_lengths)) # Start sine wave for this input case.
            stop = int(fs * ramp_time + fs * (case * slope_time + (case + 1) * amplitude_lengths)) # Stop sine wave for this input case.
            time_single_case = np.arange(0, (stop - start) / fs, 1 / fs)
            if last == False: # Do not add sines for last iteration or if threshold is met.
                input_waveforms[self.control_voltage_indices, start:stop] = input_waveforms[self.control_voltage_indices, start:stop] + np.sin(2 * np.pi * freq[:, np.newaxis] * time_single_case) * amplitude_sines[:, np.newaxis]

            # After measuring input case, ramp to next value.
            for ramps in range(len(self.control_voltage_indices)):
                input_waveforms[self.control_voltage_indices[ramps], stop:stop + int(fs * slope_time)] = np.linspace(input_waveforms[self.control_voltage_indices[ramps], stop], input_waveforms[self.control_voltage_indices[ramps], stop + int(fs * slope_time) - 1], int(fs * slope_time))
                
        # Add inputs at the correct index of the input matrix:
        for j in range(len(self.input_indices)):
            input_waveforms[self.input_indices[j], :] = np.concatenate((np.zeros(int(fs * ramp_time)), inputs[j, :], np.zeros(int(fs * ramp_time))))
        
        # Add ramping up and ramping down the voltages at start and end of iteration. TODO: integrate this in processors?
        for j in range(input_waveforms.shape[0]):
            input_waveforms[j, 0:int(fs * ramp_time)] = np.linspace(0, input_waveforms[j, int(fs * ramp_time)], int(fs * ramp_time))
            input_waveforms[j, -int(fs * ramp_time):] = np.linspace(input_waveforms[j, -int(fs * ramp_time + 1)], 0, int(fs * ramp_time))    

        return input_waveforms, ramp_mask


    def update_controls(self, epoch, outputs, y_train, data): 
        '''
            Backward step of optimization. Calculates gradient of loss w.r.t. output and gradient of output w.r.t. controls
            and combines the two to create gradient of loss w.r.t. controls.
            inputs:
                epoch: iteration number
                outputs: (masked) output from current iteration
                y_train: (masked) target values
                data: the data object. Required to as argument so that all parameters can be updated within this function.

            outputs:
                data: updated data object, including the newly updated control voltages.
        '''
        fs = self.processor_config['sampling_frequency']
        amplitude_lengths = self.waveform['amplitude_lengths'] # in seconds
  
        # Initialize placeholders for split data and gradients.
        outputs_split = np.zeros((self.waveform['input_cases'], int(fs * amplitude_lengths))) # [input cases x single case signal length]
        y_train_split = np.zeros((self.waveform['input_cases'], int(fs * amplitude_lengths))) # [input cases x single case signal length]
        IVgradients = np.zeros((self.waveform['input_cases'], self.nr_control_voltages)) # [input cases x single case signal length]
        EVgradients = np.zeros(self.nr_control_voltages)
        sign = np.zeros((self.waveform['input_cases'], self.nr_control_voltages))

        # Calculate the gradient of the cost function with respect to the output for the complete (masked) signal.
        EIgradients = self.grad_fn(outputs, y_train) 

        # Split data into different cases and compute IVgradients & EVgradients.
        for case in range(self.waveform['input_cases']):
            outputs_split[case] = outputs[int(case * fs * amplitude_lengths) : int(fs * (case + 1) * amplitude_lengths)]
            y_train_split[case] = y_train[int(case * fs * amplitude_lengths) : int(fs * (case + 1) * amplitude_lengths)]
            
            IVgradients[case] = gradient_estimator(outputs_split[case], self.processor_config["sampling_frequency"], self.waveform["frequency_factor"] * np.array(self.waveform['frequencies']), np.array(self.waveform['amplitude_sines']), self.hyperparams["phase_threshold"])
            EVgradients += np.mean(EIgradients[int(case*fs*amplitude_lengths):int((case + 1) * fs * amplitude_lengths)][:, np.newaxis] * IVgradients[case, :], axis=0)

        controls = self.optimizer(data.results["controls"][epoch], self.hyperparams['learning_rate'], EVgradients)

        # Check that new controls are within specified voltage range.
        for control_index in range(len(controls)):
            controls[control_index] = min(np.array(self.processor_config['CVrange'])[control_index, 1], controls[control_index])
            controls[control_index] = max(np.array(self.processor_config['CVrange'])[control_index, 0], controls[control_index])

        # Reset controls if outputs are clipping. TODO: This could be done more elegantly by for example only resetting a single control voltage.
        if abs(np.mean(outputs)) > self.processor_config['output_clipping_value'] * self.processor_config['amplification']:
            print("\n output clipped, resetting controls...")
            controls = np.random.random(self.nr_control_voltages) * (np.array(self.processor_config['CVrange'])[:, 1] - np.array(self.processor_config['CVrange'])[:, 0]) + np.array(self.processor_config['CVrange'])[:, 0]

        data.update({'iteration': epoch, 'IVgradients': IVgradients, 'EIgradients': EIgradients, 'EVgradients': EVgradients, 'sign': sign, 'controls': controls})
        return data
    
    def measure_best(self, inputs, targets, data):
        '''
            After optmizing the device, the best found solution is measured again, but without the sine wave perturbations.
        '''
        best_iteration = data.results['error'].argmin()
        best_controls = data.results['controls'][best_iteration]
        # create inputs without sine waves by faking that it reached the threshold value
        best_inputs, ramp_mask = self.generate_input_waves(inputs, best_controls, True)
        outputs = self.processor.get_output(best_inputs.T) [:, 0]
        error = self.loss_fn(outputs[ramp_mask][data.results['mask']], targets[data.results['mask']])
        data.update({'iteration': 0, 'best_outputs': outputs, 'best_inputs': best_inputs, 'best_error': error})


    def close(self):
        '''
            Experiments in hardware require that the connection with the drivers is closed.
            This method helps closing this connection when necessary.
        '''
        try:
            self.processor.close_tasks()
        except AttributeError:
            print('There is no closing function for the current processor configuration. Skipping.')

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from bspyalgo.utils.io import load_configs
    import bspyalgo.algorithms.gradient_on_chip.core.input_tasks as tasks
    configs = load_configs('configs/gd_on_chip/configs_template_gd_on_chip.json')
    test = OnChipGD(configs)
    # Test creating input waveforms.
    time, inputs, weights, targets = tasks.booleanLogic('AND', configs["waveform"]["amplitude_lengths"] * 4, configs["waveform"]["slope_lengths"], configs["processor"]["sampling_frequency"])
    controls = np.random.random((5))
    input_waveforms, ramp_mask = test.generate_input_waves(inputs*configs["waveform"]["input_scale"] + configs["waveform"]["input_offset"], controls)

    plt.figure()
    plt.plot(input_waveforms.T)
    plt.show()

    # Test creating data object and update controls function.
    processor = 'test'
    data = GDData(inputs, targets, configs['hyperparameters']['nr_epochs'], processor, validation_data=(None, None), mask=weights)
    data.init_arrays(configs)
    outputs = np.random.random(input_waveforms.shape[1]) + np.random.normal(0,0.1,input_waveforms.shape[1])
    data = test.update_controls(0, outputs[ramp_mask][data.results['mask']], targets[data.results['mask']], data)

    print(data.results["controls"][0])
    print(data.results["controls"][1])
    print("Test done.")



