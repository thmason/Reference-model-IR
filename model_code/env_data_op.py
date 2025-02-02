# env_data_op
# Sitai Kou
# 20240923

'''This is the all in one envelop data processing function, it takes rf data as input and use it to get envelop data'''

# %% 
# import libraries and define functions
import glob
import os
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

# envelop detection 
def env_data_get(rf_data, median_filter_size=0):
    if median_filter_size !=0: 
        rf_data_process = scipy.signal.medfilt(rf_data, kernel_size=[1,1,median_filter_size]) 
    env_data = np.abs(scipy.signal.hilbert(rf_data_process, axis=2))
    return env_data

def env_data_get_hard(rf_data, filter_size=61):
    hilbert_61 = [  0.0000,  -0.0073, -0.0000, -0.0056, 0.0000,  -0.0078, 0.0000,  -0.0104, 0.0000,
    -0.0137, 0.0000,  -0.0177, 0.0000,  -0.0227, 0.0000,  -0.0289, 0.0000,  -0.0368,
    -0.0000, -0.0472, 0.0000,  -0.0618, -0.0000, -0.0839, -0.0000, -0.1222, -0.0000,
    -0.2091, 0.0000,  -0.6356, 0,       0.6356,  -0.0000, 0.2091,  0.0000,  0.1222,
    0.0000,  0.0839,  0.0000,  0.0618,  -0.0000, 0.0472,  0.0000,  0.0368,  -0.0000,
    0.0289,  -0.0000, 0.0227,  -0.0000, 0.0177,  -0.0000, 0.0137,  -0.0000, 0.0104,
    -0.0000, 0.0078,  -0.0000, 0.0056,  0.0000,  0.0073,  -0.0000]
    hilbert_21 = [ -0.00358351, 0.00198471,  -0.01711389, 0.00345192,  -0.04015101, -0.01708274, -0.0557923,
    -0.11654848, -0.04229307, -0.59792162, 0.,          0.59792162,  0.04229307,  0.11654848,
    0.0557923,   0.01708274,  0.04015101,  -0.00345192, 0.01711389,  -0.00198471, 0.00358351]
    rf_data_process = np.zeros(rf_data.shape)
    for j in range(rf_data.shape[0]):
        for i in range(rf_data.shape[1]):
            if filter_size == 61:
                rf_data_process[j,i,:] = np.convolve(rf_data[j,i,:],hilbert_61,mode="same")
            elif filter_size ==21:
                rf_data_process[j,i,:] = np.convolve(rf_data[j,i,:],hilbert_21,mode="same")
            else:  
                print("please offer a valid filter size for hilbert")
    env_data = np.abs(rf_data_process)
    return env_data


#dB compress the data
def env_data_compress(env_data,pow=[0.3], value_reject = [5]):
    env_data_db = np.zeros(env_data.shape)
    for i in range(env_data.shape[0]):
        env_data_db[i,:,:] = 20*np.pow(env_data[i,:,:],pow[i])
        env_data_db[i,:,:]  = np.where(env_data_db[i,:,:]  < value_reject[i], 0, env_data_db[i,:,:] )
    return np.mean(env_data_db,axis = 0)


# apply TGC to the data         NOTE: I'm debating whether I should apply this to envelop data or image data, the resutls should be approximately equivalent
def env_data_tgc(env_data, TGC = [1,1,1,1,1,1,1,1]):
    f = scipy.interpolate.interp1d(np.linspace(0,len(TGC),len(TGC)), TGC,'cubic')
    tgc_interp = f(np.linspace(0,len(TGC),env_data.shape[1]))
    env_data_tgced = np.empty(env_data.shape)
    for i in range(env_data.shape[0]):
        if len(env_data.shape)==3:
            for j in range(env_data.shape[2]):
                env_data_tgced[i,:,j] = env_data[i,:,j]*tgc_interp
        else:
            env_data_tgced[i,:] = env_data[i,:]*tgc_interp
    return env_data_tgced

# interpolate the envelope data
def env_data_interpolate(rf_data,interp_factor = 2):
    x = np.linspace(0,1,rf_data.shape[0])
    y = np.linspace(0,1,rf_data.shape[1])
    X = np.linspace(0,1,rf_data.shape[0]*interp_factor)
    Y = np.linspace(0,1,rf_data.shape[1]*interp_factor)
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    rf_data_interp = scipy.interpolate.interpn((x,y), rf_data,(X,Y))
    return np.transpose(rf_data_interp)

# apply qbp filtering
def env_qbp_filter(img_data,filter_kernel):
    qbp_data = np.tile(np.zeros(img_data.shape),[filter_kernel.shape[0],1,1])
    for j in range(filter_kernel.shape[0]):
        for i in range(img_data.shape[0]):
            qbp_data[j,i,:] = np.convolve(img_data[i,:],filter_kernel[j,:],mode="same")
    return qbp_data

# plot the envelop data
def env_data_plot(env_data, loc_plot=64,fig_title='Env_image'):

    plt.subplots(1,2,figsize=(12,9), gridspec_kw={'width_ratios': [2.5,1]})
    plt.subplot(1,2,1)
    plt.imshow(np.transpose(env_data), extent=[0, 1, 0, 1.2])
    plt.xticks(np.arange(0, 0.999, step=0.249),['ch1', 'ch32','ch64','ch96','ch128'] )
    plt.yticks([])
    plt.ylabel('Depth')
    plt.title(fig_title)
    
    plt.subplot(1,2,2)
    plt.plot(env_data[loc_plot,:],np.flip(range(0, env_data.shape[1])))
    plt.xticks([0.5],[f'ch:{loc_plot}'] )
