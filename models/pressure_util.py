
import numpy as np
import cv2

SENSEL_COUNTS_TO_NEWTON = 1736   # How many force counts are approximately equal to one gram
SENSEL_PIXEL_PITCH = 0.00125    # The size of each force pixel
SENSEL_MAX_VIS = 20

def convert_counts_to_newtons(input_array):
    return input_array / SENSEL_COUNTS_TO_NEWTON

def convert_kPa_to_newtons(kPa):
    return kPa * 1000 * (SENSEL_PIXEL_PITCH ** 2)

def convert_counts_to_kPa(input_array):
    # convert to kilopascals
    force = convert_counts_to_newtons(input_array)
    pa = force / (SENSEL_PIXEL_PITCH ** 2)
    return pa / 1000

def get_pressure_kPa(pressure):
    kPa = convert_counts_to_kPa(pressure)
    return kPa

def get_force_warped_to_img(pressure,image,homography,image_size):
    force_img = get_pressure_kPa(pressure)
    force_warped = cv2.warpPerspective(force_img, homography,(image_size[1],image_size[0]) )
    return force_warped

def pressure_to_colormap(kPa, colormap=cv2.COLORMAP_INFERNO):
    # Rescale the force array and apply the colormap

    pressure_array = kPa * (255.0 / SENSEL_MAX_VIS)     # linear scaling
    pressure_array[pressure_array > 255] = 255

    force_color = cv2.applyColorMap(pressure_array.astype(np.uint8), colormap)
    return force_color


def get_force_overlay_img(pressure,image,homography,image_size, colormap=cv2.COLORMAP_OCEAN,only_force=False):
    force_warped = get_force_warped_to_img(pressure,image,homography,image_size)
    
    force_color_warped = pressure_to_colormap(force_warped, colormap=colormap)
    if only_force:
        return force_color_warped
    return cv2.addWeighted(image, 1.0, force_color_warped, 1.0, 0.0)


