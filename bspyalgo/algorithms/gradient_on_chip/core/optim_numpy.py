

def get_optimizer(configs):
    if configs['optimizer'] == 'basic_GD':
        return basic_GD
    else: print("Optimizer not recognized.")

def basic_GD(controls, learn_rate, gradients):

    return controls - learn_rate * gradients

