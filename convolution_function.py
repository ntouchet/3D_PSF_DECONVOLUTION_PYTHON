import numpy as np
from PIL import Image
from scipy.ndimage import convolve



def main():
    x_resolution = 100
    y_resolution = 100
    image_data = np.zeros((x_resolution,y_resolution))

    image_data[(x_resolution//2),(y_resolution//2)] = 1
    image_data[(x_resolution//2-1),(y_resolution//2)] = 1
    image_data[x_resolution//2,(y_resolution//2-1)] = 1
    image_data[(x_resolution//2-1),(y_resolution//2-1)] = 1

    number_of_pixles_in_kernel = 10
    kernel_start = -1
    kernel_end = 1
    x_y_kernel = np.zeros((number_of_pixles_in_kernel,1))
    z_kernel = np.zeros((number_of_pixles_in_kernel,1))
    space_between_kernel_pixels = (kernel_end-kernel_start)/number_of_pixles_in_kernel

    for index in range(number_of_pixles_in_kernel):
        z_position = kernel_start+index*space_between_kernel_pixels
        if z_position < 0:
            z_kernel[index] = 0
        else:
            z_kernel[index]=np.sinc(z_position)*np.exp(-z_position)

    for index in range(number_of_pixles_in_kernel):
        sinc = np.sinc(kernel_start+index*space_between_kernel_pixels)
        x_y_kernel[index]=sinc

    z_kernel = z_kernel.reshape(number_of_pixles_in_kernel,1)
    x_y_kernel = x_y_kernel.reshape(number_of_pixles_in_kernel,1)

    output_x = convolve(image_data, x_y_kernel, mode='reflect')
    output_xz = convolve(output_x, z_kernel.T, mode='reflect')

    output_x = convolve(image_data, x_y_kernel, mode='reflect')
    output_xy = convolve(output_x, x_y_kernel.T, mode='reflect')

    as_unsigned_int = (output_xz*255).astype(np.uint8)
    image = Image.fromarray(as_unsigned_int)
    image.save("xz_convolved.bmp")

    as_unsigned_int = (output_xy*255).astype(np.uint8)
    image = Image.fromarray(as_unsigned_int)
    image.save("xy_convolved.bmp")

    as_unsigned_int = (image_data*255).astype(np.uint8)
    image = Image.fromarray(as_unsigned_int)
    image.save("original_image.bmp")
    x_resolution = 100
    y_resolution = 100
    image_data = np.zeros((x_resolution,y_resolution))



if __name__=="__main__":
    main()
