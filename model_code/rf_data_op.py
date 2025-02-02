# rf_data_op
# Sitai Kou
# 20240923

'''This is the all in one rf data processing function'''

# %% 
# import libraries and define functions
import glob
import os
import numpy as np
import struct 
import scipy
import matplotlib.pyplot as plt


def rf_data_read(file_path, file_name1=None, file_name2=None):
    # get the file
    os.chdir (file_path)
    file_name1 = None
    file_name2 = None
    if file_name1 == None and file_name2 == None:
        preset_file_name = '*_FRAME_TYPE_BMODE_*.bin'
        Files = glob.glob(preset_file_name)
        file_name = Files[0:2]
    else:
        file_name = []
        file_name.append(file_name1)
        file_name.append(file_name2)
    
    # open 1-64 channel
    with open(file_name[0], 'rb') as file: 
        data = file.read() 
        values = struct.unpack('h' * (len(data) // 2), data)  # 'h' for 16-bit signed integer
    data_ints = np.reshape(values,[64,int(len(values)/128/64),128],order='F')
    # open 65-128 channel
    with open(file_name[1], 'rb') as file: 
        data = file.read()  
        values = struct.unpack('h' * (len(data) // 2), data)  # 'h' for 16-bit signed integer
    data_ints = np.append(data_ints,np.reshape(values,[64,int(len(values)/128/64),128],order='F'),axis=0)
    # data should be arranged in #channel * sample size * #xmit at this point

    return data_ints.astype(np.int16)

def rf_data_read_needle(file_path, file_name1=None, file_num = 1, ch=0):
    # get the file
    os.chdir (file_path)
    file_name1 = None
    if file_name1 == None:
        preset_file_name = '*_FRAME_TYPE_NEEDLE_*.bin'
        Files = glob.glob(preset_file_name)
        file_name = Files[0:2]
    else:
        file_name = []
        file_name.append(file_name1)
    
    # open 1-64 channel
    with open(file_name[file_num], 'rb') as file: 
        data = file.read() 
        values = struct.unpack('h' * (len(data) // 2), data)  # 'h' for 16-bit signed integer
    data_ints = np.reshape(values,[64,int(len(values)/128/64),128],order='F')
    
    return data_ints[ch,:,:].astype(np.int16)


def rf_data_read_npy(file_path, file_name=None, File_num = None):
    if True:
        if file_name == None:
            Files = glob.glob(os.path.join(file_path,'*_raw_rf.npy'))
        else:
            Files = glob.glob(os.path.join(file_path,'*'+file_name+'*'))
        if File_num == None:
            file_name = Files[0]
        else:
            file_name = Files[File_num]
    rf_data = np.load(file_name)
    return rf_data.astype(np.int16)
    
def rf_data_remap(raw_rf, remap ):
    if remap:
        raw_rf = raw_rf[list(remap), :, :]
    return raw_rf

# get the diagonal of the rf data for inspection only
def rf_data_diag(rf_data):
    # take the data diagonal for each pulse-echo
    img_data = np.zeros((rf_data.shape[0],rf_data.shape[1]))
    for i in range(rf_data.shape[0]):
        img_data[i,:] = rf_data[i,:,i]
    return img_data

# interpolare RF data in the depth dimension
def rf_data_interp(rf_data,interp_factor):
    # interpolating in aline
    rf_data_interp = np.zeros((rf_data.shape[0],int(rf_data.shape[1]*interp_factor), rf_data.shape[2]))
    for i in range(rf_data.shape[0]):
        for j in range(rf_data.shape[2]):
            f = scipy.interpolate.interp1d(np.linspace(0,rf_data.shape[1]-1,rf_data.shape[1]), rf_data[i,:,j],'cubic')
            rf_data_interp[i,:,j] = f(np.linspace(0,rf_data.shape[1]-1,int(rf_data.shape[1]*interp_factor)))
    return rf_data_interp

# interpolare RF data in 3D, NOTE: this is for doubling of reconstruction line density 
def rf_data_interp_double(rf_data,interp_factor):
    # interpolating in aline
    rf_data_interp = np.zeros((rf_data.shape[0]*2,rf_data.shape[1]*interp_factor, rf_data.shape[2]))
    for i in range(rf_data.shape[0]):
        for j in range(rf_data.shape[2]):
            f = scipy.interpolate.interp1d(np.linspace(0,rf_data.shape[1]-1,rf_data.shape[1]), rf_data[i,:,j],'cubic')
            rf_data_interp[i*2,:,j] = f(np.linspace(0,rf_data.shape[1]-1,rf_data.shape[1]*interp_factor))
    # interpolating in transmit. uses nearest neighbors to add extra lines of data between real data lines
    for i in range(rf_data.shape[0]-1):
        rf_data_interp[i*2+1,:,:] = (rf_data_interp[i*2,:,:] + rf_data_interp[i*2+2,:,:])/2
        
    return rf_data_interp.astype(np.int16)

def rf_data_append(rf_data,num_samples=2,loc=0):
    # appending to the end
    rf_data_zeros = np.zeros((rf_data.shape[0],num_samples,rf_data.shape[2]))
    if loc == 1:
        rf_data_appended = np.concatenate((rf_data_zeros,rf_data),axis=1)
    else:
        rf_data_appended = np.concatenate((rf_data,rf_data_zeros),axis=1)
    return rf_data_appended

def rf_data_apod(rf_data,num_samples=2,loc=0):
    # apoding the end
    if loc == 0:
        rf_data[:,-1*num_samples:,:] = 0
    else:
        rf_data[:,:num_samples,:] = 0
    return rf_data

# filter B scan Signal
def rf_data_bandpass(RFdata,cutoff_low,cutoff_high,noise_floor = 50e-4,decim_rate=2):
    # define filter, change the filter rtpe here
    filter_coeffs = scipy.signal.firwin(numtaps=61,cutoff=[cutoff_low, cutoff_high],  fs=62.5/decim_rate, pass_zero=False,
              window='blackmanharris', scale=False)
    RFdataFilter = np.empty(RFdata.shape)
    # perform filtering
    for line in range(RFdata.shape[0]):
        for ch in range(RFdata.shape[2]):
            RFdataFilter[ch,:,line] = scipy.signal.lfilter(filter_coeffs, 1, RFdata[ch,:,line])
    RFdataFilter = np.where(np.abs(RFdataFilter)<=noise_floor, 0 , np.sign(RFdataFilter)*(np.abs(RFdataFilter)-noise_floor))
    return  RFdataFilter

# plot rf data for inspection
def rf_data_plot(RFdata, loc_plot=64,fig_title='RF_image'):
    plt.subplots(1,2,figsize=(12,9), gridspec_kw={'width_ratios': [2.5,1]})
    plt.subplot(1,2,1)
    plt.imshow(np.transpose(np.abs(scipy.signal.hilbert(RFdata))), extent=[0, 1, 0, 1.2])
    plt.xticks(np.arange(0, 0.999, step=0.249),['ch1', 'ch32','ch64','ch96','ch128'] )
    plt.yticks([])
    plt.ylabel('Depth')
    plt.title(fig_title)
    
    plt.subplot(1,2,2)
    plt.plot(RFdata[loc_plot,:],np.flip(range(0, RFdata.shape[1])))
    plt.xticks([0.5],[f'ch:{loc_plot}'] )

