import math
import numpy as np



def get_typed_tex_2d(tex, xi, yi):
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tex2D#tex2d
    # return tex2D<TexType<T>::type>(tex, xi, yi);

    ### get intermediate values of a 2d array (compute weighted average of surrounding pixels)
    if float(int(xi) - xi) == 0.0 and float(int(yi) - yi) == 0.0:
        return tex[xi][yi]
    top_left = tex[math.floor(xi)][math.floor(yi)]
    top_right = tex[math.floor(xi)][math.ceil(yi)]
    bot_left = tex[math.ceil(xi)][math.floor(yi)]
    bot_right = tex[math.ceil(xi)][math.ceil(yi)]
    ud_dist = xi - math.floor(xi) # left right offset
    lr_dist = yi - math.floor(yi) # up down offset
    # i,j = ith row, jth column
    left_avg = (top_left * ud_dist + bot_left * (1-ud_dist)) / 2
    right_avg = (top_right * ud_dist + bot_right * (1-ud_dist)) / 2
    ret_val = (left_avg * lr_dist + right_avg * (1-lr_dist)) / 2
    if ret_val == 0:
        print("GET TYPED TEX 2D VAL IS 0", tex, xi, yi)
    return ret_val

def get_typed_tex_1d(tex, yi):
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tex2D#tex2d
    # return tex2D<TexType<T>::type>(tex, xi, yi);

    ### get intermediate values of a 2d array (compute weighted average of surrounding pixels)
    if float(int(yi) - yi) == 0.0:
        return tex[yi]
    left = tex[math.floor(yi)-1]
    right = tex[math.ceil(yi)-1]
    lr_dist = yi - math.floor(yi) # up down offset
    ret_val = (left * lr_dist + right * (1-lr_dist)) / 1
    return ret_val

def cubic_y(tex, y):
    u, yf = math.modf(y) # u should be fraction, yf is integer
    s0 = get_typed_tex_1d(tex, yf - 1 + 0.5)
    s1 = get_typed_tex_1d(tex, yf - 0 + 0.5)
    s2 = get_typed_tex_1d(tex, yf + 1 + 0.5)
    s3 = get_typed_tex_1d(tex, yf + 2 + 0.5)

    a0 = 0 + u * (-1 + u * (2 * u - 1))
    a1 = 2 + u * (0 + u * (-5 * u + 3))
    a2 = 0 + u * (1 + u * (4 * u - 3))
    a3 = 0 + u * (0 + u * (-1 * u + 1))

    return (s0*a0 + s1*a1 + s2*a2 + s3*a3) * 0.5


def interp1d(interp_mode, i_sample, i_receive, i_transmit, m_channel_map, m_dims, m_tex_list):  #
    i_board = 0
    i_channel = 0
    if m_channel_map != None:
        i_board = m_channel_map[i_receive].i_board
        i_channel = m_channel_map[i_receive].i_channel
    else:
        i_board = i_receive / m_dims[0]  # .num_channels
        i_channel = i_receive / m_dims[0]  # .num_channels
    i_tex = i_board * m_dims[2] + i_transmit
    # print("ITEX, IBOARD, MDIMS2, ITRANSMIT:", i_tex, i_board, m_dims[2], i_transmit)

    if interp_mode == "cubic":
        value = cubic_y(m_tex_list[int(i_tex)], i_channel, i_sample)
    else:
        print("ONLY CUBIC INTERP MODE SUPPORTED CURRENTLY")

    return value * 32767  # value of SHRT_MAX (see bte_rf_data_interp 76)
    # could be if, but it seems call always is int16_t

def create_tex_list(in_pitched):
    in_pitched = np.expand_dims(in_pitched, axis=-1)
    result = []
    m_tex_list = [] #should have len host_interp.m_num_tex = host_interp.m_dims.num_boards * host.interp.m_dims.num_transmits 212
    dims = in_pitched.shape # passed into LoadPitchedTensor
    # channels, samples, transmits, boards
    #num_frames = int(math.prod(dims) / (dims[0] * dims[1] * dims[2] *dims[3]))
    # print("NUM FRAMES:", num_frames)
    pitch = dims[0] # in Tensor.cu
    frame_size = pitch * dims[1] * dims[2] * dims[3]
    for n in range(dims[4]): # theoretically iterating thru frames
        # dev_ptr = in_pitched + n * frame_size
        dev_ptr = in_pitched[..., n]  # Slice the last dimension for the nth frame
        m_num_tex = dims[3] * dims[2]
        for i_tex in range(m_num_tex):
            # dev_ptr points at row corresponding to i_tex th board/transmit
            # interp = dev_ptr + pitch * dims[1] * i_tex
            interp = dev_ptr[:, :, i_tex // dims[3], i_tex % dims[3]]
            m_tex_list.append(interp)
        result.append(m_tex_list)
    return m_tex_list