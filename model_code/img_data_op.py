# img_data_op
# Sitai Kou
# 20240923

'''This is the all in one image data processing function, it takes beamformed data input and convert to images'''

# %% 
# import libraries and define functions
import numpy as np 
import scipy
import matplotlib.pyplot as plt
from PIL import Image
import cv2
## image operations

# convert data to gray scale uint8 image
def image_conversion(img_data,Resize_1=10,Resize_2=2):
    img_data_resize = scipy.signal.decimate(scipy.signal.decimate(img_data, Resize_1, axis=0), Resize_2, axis=1)*255.0
    US_img = np.empty([img_data_resize.shape[0],img_data_resize.shape[1],3])
    for i in range(3):
        US_img [:,:,i] = img_data_resize
    US_img = np.where(US_img < 0, 0, US_img)
    US_img = np.where(US_img > 255, 255, US_img)
    img_Out = US_img.astype(np.uint8)
    return cv2.rotate(img_Out, cv2.ROTATE_90_CLOCKWISE)

# interpolate image
def image_interpolate(image,interp_channel=2,interp_depth=2):
    image_interp = cv2.resize(image, (image.shape[0]*interp_channel, image.shape[0]*interp_depth),
        interpolation = cv2.INTER_CUBIC)
    return image_interp

# save the image
def image_save(img_data,filepath,num,typeData='US'):
    # Create an image from the uint8 data
    imageUS = Image.fromarray(img_data.astype(np.uint8))
    # Save the image
    name = typeData + '_' + num + '.tiff'
    name = filepath +"\\" + name
    imageUS.save(name)

# give the image a custom lookup table
def image_LUT(imgData,LUT= np.arange(256, dtype = np.dtype('uint8'))):
    lut = np.dstack((LUT, LUT, LUT))
    dstImage = cv2.LUT(imgData, lut)
    return dstImage

# adjusting image brightness and contrast
def image_contrast(US_img, brightness = 1, contrast = 1):
    US_img = cv2.addWeighted(US_img, contrast, np.zeros(US_img.shape, US_img.dtype), 0, brightness) 
    return US_img

# adjusting image sharpness
def image_sharpness(US_img, kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])):
    US_img = cv2.filter2D(US_img, -1, kernel)
    return US_img

# convert image to polar coordinate
def image_polar(US_img, xdc,field, param):
    append_axial = int(xdc["radius"]/(field["axial_size"]/US_img.shape[0]*field["num_samples"]))
    append_lateral = int((US_img.shape[1] * (2*np.pi-xdc["angle"])/xdc["angle"])/2)

    US_img = np.transpose(np.fliplr(US_img[:,:,0]))
    US_img = np.append(np.zeros([US_img.shape[0],append_axial]),US_img,axis=1)
    US_img = np.append(np.zeros([append_lateral,US_img.shape[1]]),US_img,axis=0)
    US_img = np.append(US_img,np.zeros([append_lateral,US_img.shape[1]]),axis=0)

    US_img = cv2.warpPolar(US_img, [US_img.shape[1]*2,US_img.shape[1]*2] ,[US_img.shape[1],US_img.shape[1]] , US_img.shape[1], cv2.WARP_INVERSE_MAP)
    US_img = cv2.rotate(US_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    US_img = US_img[int(US_img.shape[0]/2+append_axial*np.cos(xdc["angle"]/2)):,
                    int((1-np.sin(xdc["angle"]/2))*US_img.shape[0]/2):int((1+np.sin(xdc["angle"] /2))*US_img.shape[0]/2)]
    US_img_polar = np.dstack((US_img,US_img,US_img)).astype(np.uint8)
 
    return US_img_polar

def image_crop(US_img, img_length=0.05, linelength=0.06):
    US_img = US_img[:int(US_img.shape[0]*img_length/linelength),:,:]
    return US_img

# plot the image
def image_plot_linear(US_img,img_length = 0.05,lateral_length = 0.0576, fig_title='US B Scan', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    # calculate the image dimensions
    ax.imshow(US_img, cmap='gray', vmin=0, vmax=1, extent=[0, lateral_length/img_length, 0, 1])
    ax.set_title(fig_title)
    ax.set_xticks(np.arange(0, 0.999*lateral_length/img_length, step=0.249*lateral_length/img_length),['ch1', 'ch32','ch64','ch96','ch128'] )
    ax.set_xlabel('channel #')
    ax.set_yticks(np.arange(0, 0.999, step=0.249),[str(int(img_length*1.00*1e3)),
                                                str(int(img_length*0.75*1e3)),
                                                str(int(img_length*0.50*1e3)),
                                                str(int(img_length*0.25*1e3)),
                                                str(int(img_length*0.00*1e3))])
    ax.set_ylabel('depth (mm)')
    
def image_plot_curvelinear(US_img,img_length = 0.05, fig_title='US B Scan', ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    # calculate the image dimensions
    ratio = US_img.shape[0]/US_img.shape[1]
    img_width = img_length/ratio
    ax.imshow(US_img, cmap='gray', vmin=0, vmax=1, extent=[0, 1/ratio, 0, 1])
    ax.set_title(fig_title)
    ax.set_xticks(np.arange(0, 0.999/ratio, step=0.249/ratio),[str(int(img_width*0.00*1e3)),
                                                str(int(img_width*0.25*1e3)),
                                                str(int(img_width*0.50*1e3)),
                                                str(int(img_width*0.75*1e3)),
                                                str(int(img_width*1.00*1e3))])
    ax.set_xlabel('width (mm)')
    ax.set_yticks(np.arange(0, 0.999, step=0.249),[str(int(img_length*1.00*1e3)),
                                                str(int(img_length*0.75*1e3)),
                                                str(int(img_length*0.50*1e3)),
                                                str(int(img_length*0.25*1e3)),
                                                str(int(img_length*0.00*1e3))])
    ax.set_ylabel('depth (mm)')
    