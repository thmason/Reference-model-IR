
import os
import sys
import timeit
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from line_profiler import LineProfiler
from PIL import Image

sys.path.insert(0,'../model_code')

from BF_main import read_data_and_configs, run_beamformer, run_image_processing, pre_process_configs, precompute_geometry

def run_a_challenge(data_dir):
    config_path = os.path.join(data_dir, 'presets/scratch')
    raw_rf_path = os.path.join(data_dir, 'data')
    image_files = glob.glob(os.path.join(raw_rf_path, '*.png'))
    cine_files = glob.glob(os.path.join(raw_rf_path, '*cine.npy'))

    mode = 0 # 0 for b mode, 1 for needle, 2 for needle imaging, 4 for doppler?
    rf_data, preset = read_data_and_configs(raw_rf_path, config_path, mode)
    xdc, field, param = pre_process_configs(preset,mode)
    recon = precompute_geometry(xdc, field, param)
    recon_image = run_beamformer(rf_data, xdc, field, param, recon)

    num_samples = rf_data.shape[1]
    title = raw_rf_path.split('\\')[-1].split('.')[0]
    run_image_processing(recon_image, xdc, field, param, num_samples=num_samples, title = title)

    num_samples = rf_data.shape[1]
    title = data_dir
    fig_image, ax_image = plt.subplots(1,2)
    image = Image.open(image_files[0])
    print(image_files[0])
    image_array = np.array(image)
    if xdc['type'] == 0:
        image_array = image_array[41:839, 442:740, :]  # grab image from display
    else:
        image_array = image_array[183:700, 230:950, :] #curvilinear image splice
    ax_image[1].imshow(image_array)
    ax_image[1].set_title('dstMain')

    run_image_processing(recon_image, xdc, field, param, num_samples, title = title, ax=ax_image[0])


challenge_set = ['L7-4_General', 'L7-4_Harmonic', 'C5-1_General_low_att', 'C5-1 Harmonics low att']
#challenge_set = ['L7-4_General']
#challenge_set = ['C5-1_General_low_att']


profiler = LineProfiler()
profiler.add_function(run_beamformer)
#profiler.add_function(recon_process_rtbf)
profiler.add_function(precompute_geometry)
profiler.add_function(run_a_challenge)

for data_dir in challenge_set:

    data_folder = f'../Challenge_sets/{data_dir}'
    profiler.disable_by_count()
    run_a_challenge(data_folder)
    profiler.disable_by_count()

# Save the profiling results to a file
with open('profiling_results.txt', 'w') as f:
    profiler.print_stats(stream=f)
plt.show()



