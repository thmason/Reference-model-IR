# preset_parser
# Sitai Kou
# 20240920

'''This is the parameter parser for preset.pbtxt'''
# %% 
# import libraries and define functions
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from dst_toolkit.proto import beamformer_config_pb2 as beamformer_proto
from dst_toolkit.proto import sequence_config_pb2 as sequence_proto
from dst_toolkit.proto import system_config_pb2 as system_proto
from dst_toolkit.proto import table_inputs_config_pb2 as table_inputs_proto
from dst_toolkit.proto import imaging_pb2 as imaging_proto
import google

# get config using proto buffer
def config_read_pbtxt(source_path: str, proto_type):
    with open(source_path, "r") as f:
        config_txt = f.read()
    config_preset = proto_type()
    google.protobuf.text_format.Parse(config_txt, config_preset)
    return config_preset

# get data from pbtxt file
def config_load_preset(data_dir,preset_dir,mode=0):
    print(f'loading preset {preset_dir}')
    beamform_config_path = f'{preset_dir}/BeamformerConfig.pbtxt' # main
    rf_meta_data_names = glob.glob(f'{data_dir}/**.pbtxt')[mode]

    preset = {}
    preset['beamformer'] = config_read_pbtxt(beamform_config_path, beamformer_proto.BeamformerConfigSet)
    preset['metadata'] = config_read_pbtxt(rf_meta_data_names, imaging_proto.RfMetadata)
    return preset 

def config_para_parse(preset, mode = 0):

    if preset['beamformer'].config_set[mode].recon.HasField('num_dummy_lines_at_beginning'):
        num_dummy_lines = preset['beamformer'].config_set[mode].num_dummy_lines_at_beginning
    else:
        num_dummy_lines = preset['metadata'].num_dummy_lines
    # getting the transducer parameters
    xdc = {
    "type" : preset['beamformer'].config_set[mode].transducer_array.type, # 0 is linear, 1 is CL
    # geometric parameters
    "element_num": preset['beamformer'].config_set[mode].transducer_array.num_elements, # number of elements
    "pitch": preset['beamformer'].config_set[mode].transducer_array.pitch_mm/1e3, # widthe of elements [m]
    "kerf": 0e-3, # distance between elements[m] NOTE: this is unused in DSTMain
    "radius": preset['beamformer'].config_set[mode].transducer_array.radius_mm/1e3,  # transducer radius [m]
    # transmit parameters
    "xmit_focus": preset['metadata'].aperture_metadata_list[0].tx_focal_distance_mm/1e3, # the location of the transmit focusing [m]
    "start_depth": preset['metadata'].aperture_metadata_list[0].start_depth_mm/1e3, # the location of the start [m]
    "start_ray": preset['metadata'].transmit_lines_descriptor.start_ray,
    "ray_increment": preset['metadata'].transmit_lines_descriptor.ray_increment,
    "total_num_rays": preset['metadata'].transmit_lines_descriptor.total_number_of_rays,
    "num_rays": int(preset['metadata'].transmit_lines_descriptor.total_number_of_rays/preset['metadata'].transmit_lines_descriptor.ray_increment),
    "num_dummy_lines": num_dummy_lines
    }
        
    # compute time delay   for each transmit
    field = {
        # field parameters
        "c" : preset['metadata'].speed_of_sound_kmps*1e3, # m/s
        "fs": preset['metadata'].aperture_metadata_list[0].sampling_freq_mhz*1e6, # sampling rate
        # rf data metadata
        "num_samples": preset['metadata'].dimensions.samples, # number of samples
        "channels": preset['metadata'].dimensions.channels, # number of channel
        "lines": preset['metadata'].dimensions.lines, # number of xmit 
        "apertures": preset['metadata'].dimensions.apertures,
        "line_length": preset['metadata'].line_length_mm/1e3, # image recon depth[m]
    }

    param = {
        "mode" : mode,
        # post processing
        "post_process_filter_kernal":np.array(preset['beamformer'].config_set[mode].line_processor.post_process_arm.filter_kernel),
        # reconstruction
        "recon_time_0":preset['beamformer'].config_set[mode].recon.time0_us,
        "start_depth":preset['metadata'].aperture_metadata_list[0].start_depth_mm,
        "recon_cf":preset['beamformer'].config_set[mode].recon.recon_center_freq_mhz*1e6,
        "recon_tx_width_wavelength":int(preset['beamformer'].config_set[mode].recon.tx_width_wavelength),
        "recon_tx_fnum":preset['beamformer'].config_set[mode].recon.tx_f_number,
        "recon_rx_fnum":preset['beamformer'].config_set[mode].recon.rx_f_number,
        "double_recon_line_density":preset['beamformer'].config_set[mode].recon.double_recon_line_density,
        "pulse_inversion_harmonics":preset['beamformer'].config_set[mode].recon.pulse_inversion_harmonics
    } 
    
    param["filter_kernel"] = np.zeros((len(preset['beamformer'].config_set[mode].line_processor.qbp_arms),len(preset['beamformer'].config_set[mode].line_processor.qbp_arms[0].filter_kernel)))
    param["filter_power"] = np.zeros(len(preset['beamformer'].config_set[mode].line_processor.qbp_arms))
    param["filter_reject_x"] = np.zeros(len(preset['beamformer'].config_set[mode].line_processor.qbp_arms))
    for i in range(len(preset['beamformer'].config_set[mode].line_processor.qbp_arms)):
        param["filter_kernel"][i,:] = np.array(preset['beamformer'].config_set[mode].line_processor.qbp_arms[i].filter_kernel)
        param["filter_power"][i] = preset['beamformer'].config_set[mode].line_processor.qbp_arms[i].power
        param["filter_reject_x"][i] = preset['beamformer'].config_set[mode].line_processor.qbp_arms[i].reject_x
    
    return xdc, field, param


def config_para_process(xdc, field, param):
    # xdc parameters
    xdc["element_width"] = xdc["pitch"]+xdc["kerf"] # actual lateral size of elements
    xdc["width"] = xdc["element_width"]*(xdc["element_num"]-1) # full width of the elements
    if xdc ["type"] == 1:
        xdc["angle"] = xdc["width"] /xdc["radius"]   # the angle in degrees is xdc["angle"]/np.pi*180
    
    # field parameters
    field["axial_size"] = field["c"]/field["fs"]/2 # actual axial size of samples
    
    # reconstruction parameters
    param['recon_wavelength'] = field["c"]/param['recon_cf'] # wavelength used to define pixels
    # recon_tx_width:  xmit beamwidth used in reconstruction
    param['tx_width_mm'] = max(0.0, param["recon_tx_width_wavelength"]*param['recon_wavelength'])
    # recon_depth_spacing: distance between pixels in depth direction
    param['recon_depth_spacing'] = param['recon_wavelength']/4/2 # 1/8 wavelength spacing
    # recon_num_depth:  number of pixels in depth direction
    param['recon_num_depth'] = int(field["line_length"]/param['recon_depth_spacing'])
    # recon_num_length:  number of pixels in lateral direction
    if param["double_recon_line_density"]:
        param['num_tx_lines'] = 2*field['lines']-1
        param['ray_increment'] = xdc['ray_increment']/2
    else:
        param['num_tx_lines'] =  field['lines']
        param['ray_increment'] = xdc['ray_increment']

    param['num_rx_lines'] = param['num_tx_lines']
        
    # populate with default parameters
    param["rcv_delay"] = 0 
    param["tx_delay"] = 0
    param["no_rcv_beamforming"] = 0
    param["no_tx_limiter"] = 0
    param["no_rcv_limiter"] = 0
    
    return xdc, field, param
