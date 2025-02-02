# reference model beamformer for DSTMain
# Sitai Kou
# last edit: 20241011

"this is the all in one control for reconstruction"

# %% 
# import packages and functions
import os
import sys
import timeit
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import easygui
import pickle 

sys.path.insert(0,'C:\\Codes\\github\\signal_processing_dev\\dstMain_reference_model')

from preset_parser import config_load_preset, config_para_parse, config_para_process
from rf_data_op import rf_data_read,rf_data_read_npy,rf_data_remap, rf_data_bandpass,rf_data_interp,rf_data_append, rf_data_apod, rf_data_diag, rf_data_plot, rf_data_interp_double
#from beamformer_calc_rtbf import recon_compute_geometry, recon_field_pre_compute, recon_process_rtbf
from env_data_op import env_data_get, env_data_get_hard, env_data_compress, env_qbp_filter,env_data_interpolate
from img_data_op import image_conversion, image_polar,image_crop, image_plot_linear, image_plot_curvelinear, image_interpolate, image_LUT, image_contrast, image_sharpness

from beamformer import recon_compute_geometry, recon_field_pre_compute, recon_process_rtbf

GPU = False

if GPU:
# import GPU items
    from beamformer_calc_rtbf_GPU import recon_process_rtbf

def read_data_and_configs(raw_rf_path, config_path, mode=0):
    print("grabbing data and config")
    preset = config_load_preset(raw_rf_path,config_path)

    # rf_data = rf_data_read(file_path) # use this to read bin files
    rf_data = rf_data_read_npy(raw_rf_path, file_name=None)  # use this to read npy files
    if preset['beamformer'].config_set[0].recon.pulse_inversion_harmonics: #determine if pulse inversion harmonics
        rf_data = rf_data[:, 0:int(rf_data.shape[1] / 2), :] + rf_data[:, int(rf_data.shape[1] / 2):, :]
    rf_data = rf_data_remap(rf_data, list(preset["beamformer"].config_set[0].pic_sub_array.channel_map))
    # rf_data_plot(rf_data_diag(rf_data), loc_plot=64,fig_title='P/E image')# plot the rf P/E for inspection
    
    # data apodization, zeros the last two points for signal artifact removal
    rf_data = rf_data_apod(rf_data, 2, loc=0)
    # data appending, add two zeros at the end.  Beamformer will use this sample when delay and sum points beamformer to out of range data
    rf_data = rf_data_append(rf_data, 2, loc=0)
        # interpolates RF data in the time direction.
    if preset['beamformer'].config_set[mode].recon.double_recon_line_density:
        rf_data = rf_data_interp_double(rf_data,1)
    else:
        rf_data = rf_data_interp(rf_data,1)
        
    return rf_data, preset    

def pre_process_configs(preset, mode=0):
    # reformat config into
    # xdc:  transducer geometries
    # field:  acoustic field parameters
    # param:  other miscellaneous parameters
    xdc, field, param = config_para_parse(preset, mode)
    xdc, field, param = config_para_process(xdc, field, param)
    # NOTE: optinoal customizatomp of reconstruction paramater here
    # param["tx_delay"] = 0.25e-6; param["rcv_delay"] = 0.25e-6 
        
    return xdc, field, param

def precompute_geometry(xdc, field, param):
    # generate calculated parameters and add to existing config stuctures, params broken out to facilitate parameter optimization
    # pre compute reconstruction field geometry and limiters
    print('precomputing geometry...')
    recon = recon_compute_geometry(xdc, field, param)  # pre-compute reconstruction field
    print('precomputing field...')
    tic = timeit.default_timer()
    recon = recon_field_pre_compute(recon, xdc, field, param)
    toc = timeit.default_timer()
    print(f"precompute took {toc - tic} seconds ")
    return recon

def run_beamformer(rf_data, xdc, field, param, recon):
    # apply lookup tables based on precomputed geometry
    # recon image is of size lines X channels X number of pixels
    tic = timeit.default_timer()
    print('performing delay and sum...')
    recon_image = recon_process_rtbf(rf_data, xdc, recon, field, param)
    toc = timeit.default_timer()
    print(f"beamforming took {toc - tic} seconds ")

    return recon_image


def run_image_processing(recon_image, xdc, field, param, num_samples, dynamic_range=150, title = 'image', ax = None):

# %% 
    # sum the data through transmits
    # recon_image_sum = np.sum(recon_image, axis=1) / 2 ** 15
    # recon_image_sum = env_data_interpolate(recon_image_sum, interp_factor=3)
    
    print("getting envelope")
    # get envelop and compress
    recon_image_qbp = env_qbp_filter(recon_image, filter_kernel=np.array(param["filter_kernel"]))
    
    # recon_image_qbp[:, :,-4:] = 0;recon_image_qbp[:, :,:4] = 0 # apodize it to prevent hilbert transform artfact
    # recon_US_img_data = env_qbp_filter(recon_US_img_data, filter_kernal=bf_config["line_processor"]["qbp_arms_2"]["filter_kernel"])
    # recon_image_qbp_env = env_data_get(recon_image_qbp, median_filter_size=1) + 1e-12
    recon_image_qbp_env = env_data_get_hard(recon_image_qbp) +1e-12
    recon_US_img_dB = env_data_compress(recon_image_qbp_env, pow=param["filter_power"], value_reject=param["filter_reject_x"])
    
    print("plotting for display")
    
    # # add db compression
    # recon_US_img_dB = np.log10(recon_US_img_dB)*(dynamic_range/20)
    # recon_US_img_dB = np.where(recon_US_img_dB<1, 0 , recon_US_img_dB)
    
    scale=np.percentile(recon_US_img_dB,99.9)
    US_img = image_conversion(recon_US_img_dB/scale, Resize_1=1, Resize_2=1)  # convert to uint8
    US_img = image_crop(US_img, img_length=field["axial_size"] * num_samples,
                        linelength=field["line_length"])  # crop basedon available signal length
    # adding image manipulation for better display
    # US_img = image_interpolate(US_img,interp_channel=4,interp_depth=1) # interpolate the image
    # US_img = image_LUT(US_img,LUT = np.linspace(0,255,256).astype('uint8') ) # use LUT
    # US_img = image_contrast(US_img, brightness = 1, contrast = 1)
    # US_img = image_sharpness(US_img, kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
    
    if xdc["type"] == 1:
        US_img = image_polar(US_img, xdc, field, param)  # polar convert the image
        image_plot_curvelinear(np.fliplr(US_img),  
                               img_length=field["axial_size"] * num_samples + (
                                           xdc["radius"] * (1 - np.cos(xdc["angle"] / 2))),  # length of image, [m]
                               fig_title='US B Scan', ax = ax)
    else:
        image_plot_linear(np.fliplr(US_img),
                          img_length=field["axial_size"] * num_samples,  # length of image, [m]
                          lateral_length=xdc['element_num'] * xdc["element_width"],  # width of image, [m]
                          fig_title='US B Scan', ax = ax)

    plt.savefig(f'{title}.png'); plt.show()
    #plt.show()

# %% 
if __name__ == "__main__":

    data_dir = 'D:\\SK\\20241107 challenge dataset for reference model\\L7-4_General 2'
    # data_dir = easygui.diropenbox()
    config_path = os.path.join(data_dir, 'presets\\scratch')
    raw_rf_path = os.path.join(data_dir, 'data')

    mode = 0 # 0 for b mode, 1 for needle, 2 for needle imaging, 4 for doppler?
    rf_data, preset = read_data_and_configs(raw_rf_path, config_path, mode)

    if False:
        with open('recon_parameters.pkl', 'rb') as f:
            xdc, field, param, recon = pickle.load(f)
    else:
        xdc, field, param = pre_process_configs(preset,mode)
        recon = precompute_geometry(xdc, field, param)        
    if True:
        with open('D:\\recon_parameters.pkl', 'wb') as f:
            pickle.dump([xdc, field, param, recon], f)
            
    recon_image = run_beamformer(rf_data, xdc, field, param, recon)
    num_samples = rf_data.shape[1]
    title = raw_rf_path.split('\\')[-1].split('.')[0]
    run_image_processing(recon_image, xdc, field, param, 
                         num_samples=num_samples, title = title)
    #plt.show()



