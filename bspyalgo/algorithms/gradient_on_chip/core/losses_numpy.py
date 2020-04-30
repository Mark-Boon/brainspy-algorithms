import numpy as np


def choose_loss_function(loss_fn_name):
    '''Gets the loss function used in GD from the module losses.
    The loss functions must take two arguments, the outputs of the black-box and the target
    and must return a torch array of scores of size len(outputs).
    '''
    if loss_fn_name == 'MSE':
        return MSE_loss
    elif loss_fn_name == 'NMSE':
        return NMSE_loss
    elif loss_fn_name == 'cor_sigmoid':
        return cor_sigmoid_loss
    else:
        raise NotImplementedError(f"Loss function {loss_fn_name} is not recognized!")

def choose_grad_function(loss_fn_name):
    '''Gets the grad function used in GD from the module losses_numpy.
    The grad functions must take two arguments, the outputs of the black-box and the target
    and must return a torch array of scores of size len(outputs).
    '''
    if loss_fn_name == 'MSE':
        return MSE_grad
    elif loss_fn_name == 'NMSE':
        return NMSE_grad
    elif loss_fn_name == 'cor_sigmoid':
        return cor_sigmoid_grad
    else:
        raise NotImplementedError(f"Loss function {loss_fn_name} is not recognized!")

def MSE_loss(x, t):
    return np.sum(((x - t))**2)/len(t)

def MSE_grad(x, t):  
    return 0.5 * (x - t) 

def NMSE_loss(x, t):
        return np.sum(((x - t))**2)/((max(x)-min(x))* len(t))

def NMSE_grad(x, t):
    ''' Calculates the normalized mean squared error loss given the gradient of the 
    output w.r.t. the input voltages. This function calculates the error
    for each control separately '''      
    return 0.5 * (x - t) / (max(x)-min(x))

def cor_grad(x, t):
    x_min_m = x - np.mean(x)
    t_min_m = t - np.mean(t)
    num = np.mean(x_min_m * t_min_m)     # numerator of corr
    denom = np.std(x) * np.std(t)        # denominator of corr    
    d_corr = ((t_min_m)/len(t_min_m) * denom - num * (x_min_m/len(x_min_m))/np.sqrt(np.mean(x_min_m**2)) * np.sqrt(np.mean(t_min_m**2))) / (denom**2)     
    return -d_corr # '-' sign because our corr is actually 1 - corr

def cor_sigmoid_loss(x, t):
    corr = np.mean((x-np.mean(x))*(t-np.mean(t)))/(np.std(x)*np.std(t)+1E-12)
    x_high_min = np.min(x[(t > max(t)-0.05)]) 
    x_low_max = np.max(x[(t < max(t)-0.05)])
    sigmoid = 1/(1 +  np.e**(-(x_high_min - x_low_max -5)/3)) + 0.05
    return (1.1 - corr) / sigmoid  
        
def cor_sigmoid_grad(x, t):
    corr = np.mean((x-np.mean(x))*(t-np.mean(t)))/(np.std(x)*np.std(t)+1E-12)
    d_corr = cor_grad(x, t)  
    
    x_high_min = np.min(x[(t > max(t)-0.05)]) 
    x_low_max = np.max(x[(t > max(t)-0.05)])
    
    sigmoid = 1/(1 +  np.e**(-(x_high_min - x_low_max -5)/3)) +0.05
    d_sigmoid = sigmoid*(1-sigmoid)
    
    return (d_corr * sigmoid - ((x == x_high_min).astype(int) - (x == x_low_max).astype(int)) * d_sigmoid * (1.1 - corr)) / sigmoid **2 