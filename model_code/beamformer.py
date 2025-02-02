# beamformer_calc
# Sitai Kou
# 20240923

'''This is the all in one envelop data processing function, 
it takes data from beamformer preset and spits out a signal delayed by delay table'''

# %% 
# import libraries and define functions
import sys
import numpy as np 
import scipy
import copy
import matplotlib.pyplot as plt
from interpolator import cubic_y
from scipy.interpolate import CubicHermiteSpline

use_interpolator = False

# calculations based on dst_focus, this grid is contructed based on the assumtion that xdc and recon field align at the first element
# axial: ultrasound propagation direction, lateral/azimuthal: transducer array direction

def interpolate_valid_samples(data, samples):
    results = []
    diff_data = list(np.diff(data))
    diff_data.append(diff_data[-1])
    spline = CubicHermiteSpline(np.arange(0, len(data)), data, diff_data)
    interpolated_x = np.linspace(0,4000,4066*40)
    interpolated_y = [spline(i) for i in interpolated_x]

    # use dstmain interpolator
    dstmain_interpolation = [cubic_y(data, i) for i in interpolated_x]

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(data)), data, label = 'data')
    ax.plot(interpolated_x, interpolated_y, label='cubic interpolator')
    ax.plot(interpolated_x, dstmain_interpolation, label = 'dst_main interpolation')
    ax.legend()
    plt.show()


    for sample in samples:
        data_result =  cubic_y(data,sample)
        print(f'sample:  {sample}')
        print(f'closest data point {data[int(sample)]}')
        min_index = int(np.max((0,sample-3)))
        max_index = int(np.min((sample+3, len(data)-1)))
        data_vec = data[min_index:max_index+1]
        data_vec_diff = list(np.diff(data_vec))
        data_vec_diff.append(data_vec_diff[-1])
        print([f'{i:.3f}' for i in data_vec])
        print(f'data result {data_result:.3f}')

        print(f'spline result {spline(sample)}')



        print()
        results.append(data_result)
    return results

def recon_compute_geometry(xdc, field, param):
    """
    precomputing geometries for transducer element locations, virtual point sources, and pixels
    Parameters: xdc, field, param
    """

    recon = {}
    if xdc["type"] == 0: # linear geometry

        #calculate rx_origin
        total_aperture = xdc['pitch'] * (xdc['element_num']-1)
        single_ray_step = (total_aperture / (xdc['total_num_rays'] - 1))
        step = single_ray_step * param['ray_increment']
        start_x = -total_aperture * 0.5 + single_ray_step * xdc['start_ray']
        rx_origin = start_x + np.array([i*step for i in range(param['num_rx_lines'])])

        #calculate tx origin
        tx_origin = start_x + np.array([i*step for i in range(param['num_tx_lines'])])

        # calculate transducer element locations [num_elem, 3]
        total_length = total_aperture # maintaining variable for DSTmain similarity
        start_x = -total_length * 0.5
        element_positions = start_x + np.array([i*xdc['pitch'] for i in range(xdc['element_num'])])

        # calculate 3d element position matrix
        recon_tx_xdc_grid_lateral = element_positions
        recon_tx_xdc_grid_axial = np.zeros(recon_tx_xdc_grid_lateral.shape)
        recon_tx_xdc_grid_elev = np.zeros(recon_tx_xdc_grid_lateral.shape)
        recon["tx_xdc_grid_Vec"] = np.squeeze(np.dstack((recon_tx_xdc_grid_axial,recon_tx_xdc_grid_lateral,recon_tx_xdc_grid_elev)))  # Tx location grid (1D line *3)
        
        # calculate virtual point source locations, [num_xmit, 3]
        recon_vps_grid_lateral = tx_origin
        recon_vps_grid_axial = np.ones (recon_vps_grid_lateral.shape)*(xdc["xmit_focus"])
        recon_vps_grid_elev = np.zeros(recon_vps_grid_lateral.shape)
        recon["vps_grid_Vec"] = np.squeeze(np.dstack((recon_vps_grid_axial,recon_vps_grid_lateral,recon_vps_grid_elev)))  # Tx location grid (1D line *3)

        # calculate the pixel grid [recon_num_depth, recon_num_length, 3]
        recon["px_grid_axial"] = np.arange(0,param['recon_num_depth'])*param['recon_depth_spacing']
        recon['px_grid_lateral'] = rx_origin

        recon_px_grid_axial2D, recon_px_grid_lateral2D = np.meshgrid(recon["px_grid_axial"] ,recon['px_grid_lateral'])
        recon_px_grid_elev2D= np.zeros(recon_px_grid_axial2D.shape) 
        recon["px_grid_Vec"] = np.dstack((recon_px_grid_axial2D,recon_px_grid_lateral2D,recon_px_grid_elev2D))

        # Rx location grid (1D channel *3)
        recon_rcv_xdc_grid_lateral = copy.copy(recon_tx_xdc_grid_lateral)
        recon_rcv_xdc_grid_axial = np.zeros(recon_tx_xdc_grid_lateral.shape)
        recon_rcv_xdc_grid_elev = np.zeros(recon_tx_xdc_grid_lateral.shape)
        recon["rcv_xdc_grid_Vec"] = np.squeeze(np.dstack((recon_rcv_xdc_grid_axial,recon_rcv_xdc_grid_lateral,recon_rcv_xdc_grid_elev)))  # rcv location grid (1D line *3)     

        # calculate TX_range
        # a 2* len(rx_origin) array giving a start and stop index for valid tx lines for each rx line
        recon['tx_range'] = []
        for ind, rx in enumerate(rx_origin):
            # find tx_indices within range
            in_range_tx = np.where(np.abs(tx_origin - rx) <= (param['tx_width_mm'] / 2))[0]
            # remove dummy lines
            in_range_tx = [i for i in in_range_tx if i >= xdc['num_dummy_lines']]
            recon['tx_range'].append(in_range_tx)

    elif xdc["type"] == 1: # CueveLinear geometry
        # calculate transducer element locations  
        theta = np.linspace(-xdc['angle']/2, xdc['angle']/2, field['channels'])
        
        recon_tx_xdc_grid_lateral = xdc["radius"] * np.sin(theta)
        recon_tx_xdc_grid_axial = xdc["radius"] * np.cos(theta)
        recon_tx_xdc_grid_elev = np.zeros(recon_tx_xdc_grid_lateral.shape)
        recon["tx_xdc_grid_Vec"] = np.squeeze(np.dstack((recon_tx_xdc_grid_axial, recon_tx_xdc_grid_lateral,recon_tx_xdc_grid_elev))) # Tx location grid (1D line *3) 
        
        # calculate virtual point source locations
        recon_vps_grid_lateral = (xdc["radius"]+xdc["xmit_focus"]) * np.sin(theta)
        recon_vps_grid_axial = (xdc["radius"]+xdc["xmit_focus"]) * np.cos(theta)
        recon_vps_grid_elev = np.zeros(recon_vps_grid_lateral.shape)
        recon["vps_grid_Vec"] = np.squeeze(np.dstack((recon_vps_grid_axial,recon_vps_grid_lateral,recon_vps_grid_elev))) # VPS location grid (1D line *3)

        # calculate the pixel grid
        param['recon_num_length'] = 128
        recon_px_grid_axial2D = np.zeros((param['recon_num_length'],param['recon_num_depth']))
        recon_px_grid_lateral2D = np.zeros((param['recon_num_length'],param['recon_num_depth']))  
        recon_px_grid_axial = xdc["radius"]+np.arange(0,param['recon_num_depth'])*param['recon_depth_spacing']

        for i in range(param['recon_num_depth']):
            recon_px_grid_lateral2D[:,i] = (recon_px_grid_axial[i]) * np.sin(theta)
            recon_px_grid_axial2D[:,i] = (recon_px_grid_axial[i]) * np.cos(theta)
        recon_px_grid_elev2D = np.zeros(recon_px_grid_axial2D.shape)# elevation grid for handling mixed array      

        # generate full Vec pixel location
        recon["px_grid_Vec"] = np.dstack((recon_px_grid_axial2D,recon_px_grid_lateral2D,recon_px_grid_elev2D))# pixel grid Vec (2D plane *3)
        recon["px_grid_Vec_linear"] = np.reshape(recon["px_grid_Vec"],[param['recon_num_length']*param['recon_num_depth'],3], order='F') # reshape to ((lateral)*axial) *3

        # Rx location grid (1D channel *3)
        theta = np.linspace(-xdc['angle']/2, xdc['angle']/2, field["channels"])
        recon_rcv_xdc_grid_lateral = xdc["radius"] * np.sin(theta)
        recon_rcv_xdc_grid_axial = xdc["radius"] * np.cos(theta)
        recon_rcv_xdc_grid_elev = np.zeros(recon_tx_xdc_grid_lateral.shape)
        recon["rcv_xdc_grid_Vec"] = np.squeeze(np.dstack((recon_rcv_xdc_grid_axial,recon_rcv_xdc_grid_lateral,recon_rcv_xdc_grid_elev)))  # rcv location grid (1D line *3)

    if False : #recon_needle geometry
        pass

    return recon

def recon_field_apod(window_num,window_size): # calculate apodization
    if window_num == 0:
        recon_window = scipy.signal.windows.hamming(window_size)
    elif window_num == 1:
        recon_window = scipy.signal.windows.hann(window_size)
    elif window_num == 2:
        recon_window = scipy.signal.windows.blackmanharris(window_size)
    elif window_num == 3:
        recon_window = scipy.signal.windows.blackmanharris(window_size)
        recon_window = np.roll(recon_window,int(window_size/4.5))+np.roll(recon_window,-int(window_size/4.5))
    else: 
        recon_window = np.ones(window_size)
    return recon_window

def ComputeUnscaledWindowWeight(x, x_range):
    if x_range == 0:
        return 0
    xx = x/x_range + 0.5
    if xx < 0 or xx > 1:
        return 0
    return 1

def recon_field_pre_compute(recon, xdc, field, param, skip_apod = False):
    """
    Pre-compute delays abd boolean matrixes of field limiters for transmit and recieve apodication
    limiters are boolean matrix of size channel X lines X pixel_depth
    ----------
    input: recon, xdc, field, param
    Returns : updated recon file
    -------

    """
    print('pre computing delays, limiters, and apodization...')

    c_inv = 1 / field["c"]

    # calculate vectors from virtual point source to pixel locations
    # reshape vps for broadcast
    vps_for_broadcast = recon["vps_grid_Vec"].reshape(-1, 1, 1, 3)
    # add vps axis for broadcast and subtract
    pixel_to_vps_vec = recon["px_grid_Vec"][np.newaxis, :, :, :] - vps_for_broadcast

    # calculate vectors from channels source to pixel locations
    # reshape xdc for broadcast
    xdc_for_broadcast = recon["rcv_xdc_grid_Vec"].reshape(-1, 1, 1, 3)
    # add vps axis for broadcast and subtract
    px_grid_vec_for_broadcast = recon["px_grid_Vec"][np.newaxis, :, :, :]
    pixel_to_xdc_vec = px_grid_vec_for_broadcast - xdc_for_broadcast

    # calculate delay from beam center to VPS

    recon['T_vps'] = (param["recon_time_0"] +
                      xdc["xmit_focus"] * c_inv
                      -2 * xdc["start_depth"] * c_inv)

    # calculate delays (s)
    recon['pixel_to_xdc_tof'] = np.linalg.norm(pixel_to_xdc_vec, axis=3) * c_inv
    recon['pixel_to_vps_tof'] = np.linalg.norm(pixel_to_vps_vec, axis=3) * c_inv

    # calculate angles
    recon['pixel_to_xdc_angle'] = np.abs(np.arctan2(pixel_to_xdc_vec[:,:,:,1],pixel_to_xdc_vec[:,:,:,0]))

    # calculate limiters
    f_num_pad = 1 / (c_inv * param["recon_cf"])

    #calc TX limiter
    recon['pv_z'] = (pixel_to_vps_vec[:, :, :, 0])
    pv_xy_vec = pixel_to_vps_vec[:, :, :, [1, 2]]
    recon['pv_xy'] = np.linalg.norm(pv_xy_vec, axis=3)
    tx_f_num = 0.5 * (f_num_pad + np.abs(recon['pv_z'])) / recon['pv_xy']
    recon["tx_limiter"] = np.where(tx_f_num >= param["recon_tx_fnum"], 1, 0)

    # calc RX limiter
    recon['pr_z'] = np.abs(pixel_to_xdc_vec[:, :, :, 0])
    pr_xy_vec = pixel_to_xdc_vec[:, :, :, [1, 2]]
    recon['pr_xy'] = np.linalg.norm(pr_xy_vec, axis=3)
    rx_f_num = 0.5 * (f_num_pad + np.abs(recon['pr_z'])) / recon['pr_xy']
    recon["rcv_limiter"] = np.where(rx_f_num >= param["recon_rx_fnum"], 1, 0)

    if not skip_apod:

        vectorized_window_weight = np.vectorize(ComputeUnscaledWindowWeight)
        recon['apod'] = vectorized_window_weight(recon['pr_xy'] * param["recon_rx_fnum"], recon['pr_z'])

    return recon


# conduct delay and sum based on rf and pre-processed data
def recon_process_rtbf(rf_data,xdc,recon,field,param):
    """
    calculates the delay, linearize operation and perform lookup
    Parameters
    ----------
    rf_data, recon, field, param

    Returns : recon_image_sum
    -------
    """
    print('performing delay and sum')

    # perform delay and sum
    ##rtbf_range = np.arange(-int(param['tx_width_mm'] / 2),
    #                       int(param['tx_width_mm'] / 2) + 1)

    recon_image = np.zeros(recon["px_grid_Vec"][:, :, 0].shape)
    channels = list(range(field['channels']))

    for channel in channels:
        print('beamforming line {}'.format(channel))

        # get RX_delays
        rx_delays = np.copysign(recon['pixel_to_xdc_tof'][channel, :, :],
                                recon['pr_z'][channel, :, :])
        for tx_ind in recon['tx_range'][channel]:
            tx_delays = np.copysign(recon['pixel_to_vps_tof'][tx_ind, channel, :],
                                    recon['pv_z'][tx_ind, channel, :])
            tx_delays = tx_delays[np.newaxis, :]

            # total delays for each pixel
            total_delay = recon['T_vps'] + rx_delays + tx_delays
            if use_interpolator:
                samples = (total_delay * field['fs']).astype('float')
            else:
                samples = (total_delay * field['fs']).astype('int16')
            # data lookup
            rf_data_lookup = np.zeros(samples.shape)
            for channel_ind in range(field['channels']):
                # ensure sample is within range of raw_rf_data
                valid_samples = np.where(samples[channel_ind, :] < rf_data.shape[1]-3)
                if not use_interpolator:
                    rf_data_lookup[channel_ind, valid_samples] = rf_data[
                        channel_ind, samples[channel_ind, valid_samples], tx_ind]
                else:
                    rf_data_lookup[channel_ind, valid_samples] = interpolate_valid_samples(rf_data[channel_ind,:,tx_ind], samples[channel_ind,valid_samples].squeeze())


            rf_data_lookup = rf_data_lookup * recon['apod'][channel, :, :]
            eligible_pixels = np.where(
                recon["rcv_limiter"][channel, :, :] & recon["tx_limiter"][tx_ind, channel, :][np.newaxis, :])
            # zero out ineligible pixels
            rf_data_for_sum = np.zeros(rf_data_lookup.shape)
            rf_data_for_sum[eligible_pixels] = rf_data_lookup[eligible_pixels]

            recon_image[channel, :] += np.sum(rf_data_for_sum, axis=0)

    recon_image = recon_image/ 2**15

    return recon_image


def recon_Vec_geometry_inspect(matrix_Vec):
    ax = plt.axes(projection='3d')
    if len(matrix_Vec.shape) == 2:
        ax.plot3D(matrix_Vec[:,0],matrix_Vec[:,1],matrix_Vec[:,2],'green')
        ax.set_title("points in 3D")
    elif len(matrix_Vec.shape) == 3 :
        ax.plot_wireframe(matrix_Vec[:,:,0], matrix_Vec[:,:,1], matrix_Vec[:,:,2], color ='green')
        ax.set_title("surface in 3D")
    else: 
        sys.exit("geometry not supported")
    ax.set_xlabel('axial location[m]')
    ax.set_ylabel('azimuthal location[m]')
    ax.set_zlabel('elevation location[m]')
    # plt.show()


# %%
