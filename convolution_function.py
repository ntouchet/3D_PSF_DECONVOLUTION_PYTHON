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
    kernel = np.zeros((number_of_pixles_in_kernel,1))
    space_between_kernel_pixels = (kernel_end-kernel_start)/number_of_pixles_in_kernel

    for index, pixel in enumerate(kernel):
        pixel = np.sinc(kernel_start+index*space_between_kernel_pixels)
        kernel[index]=pixel

    kernel = kernel.reshape(10,1)

    output_x = convolve(image_data, kernel, mode='reflect')
    output_both = convolve(output_x, kernel.T, mode='reflect')

    as_unsigned_int = (output_both*255).astype(np.uint8)
    image = Image.fromarray(as_unsigned_int)
    image.save("convolved.bmp")

    as_unsigned_int = (image_data*255).astype(np.uint8)
    image = Image.fromarray(as_unsigned_int)
    image.save("original_image.bmp")

if __name__=="__main__":
    main()
