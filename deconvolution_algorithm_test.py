import numpy as np
import matplotlib.pyplot as plt

# create 125 points to place in sample 
def get_centers(image_dimensions):

    rng = np.random.default_rng()
    centers = np.zeros((125,3),dtype="int");

    for i in range(0,5):

        x_unit_length = round(image_dimensions[0]/5)
        x_region_depth = i*x_unit_length
        x_reals = rng.poisson(x_region_depth+round(x_unit_length/2),25)
        x_reals[x_reals>image_dimensions[0]] = image_dimensions[0]

        for j in range(0,5):

            y_unit_length = round(image_dimensions[1]/5)
            y_region_depth = j*y_unit_length
            y_reals = rng.poisson(y_region_depth+round(y_unit_length/2),5)
            y_reals[y_reals>image_dimensions[1]] = image_dimensions[1]

            for k in range(0,5):
                
                z_unit_length = round(image_dimensions[2]/5)
                z_region_depth = k*z_unit_length
                z_reals = rng.poisson(z_region_depth+round(z_unit_length/2),1)
                z_reals[z_reals>image_dimensions[2]] = image_dimensions[2]

                center_of_voxel = np.array([x_reals[j*5+k], y_reals[k], z_reals[0]],dtype="int")
                

                centers[25*i + 5*j + k] = center_of_voxel

    return centers

def get_dot_intensities(
    image_dimensions, dot_center
):
    rng = np.random.default_rng()
    intensity_map_3D = np.zeros(image_dimensions)

    max_intensity = 0

    for i in range(
        int(max(dot_center[0]-4,0)),
        int(min(dot_center[0]+5,image_dimensions[0]))
    ):
        for j in range(
            int(max(dot_center[1]-4,0)),
            int(min(dot_center[1]+5,image_dimensions[1]))
        ):
            for k in range(
                int(max(dot_center[2]-4,0)),
                int(min(dot_center[2]+5,image_dimensions[2]))
            ):
                
                distance = np.sqrt(sum(np.square(
                    (np.array([dot_center[0],dot_center[1],dot_center[2]])-np.array([i,j,k]))*np.array([1,1,image_dimensions[1]/image_dimensions[2]])
                )))

                intensity_coefficient = np.exp(-0.5*np.square(distance))
                #voxel_intensity = rng.uniform(0,1)*intensity_coefficient
                voxel_intensity = 1*intensity_coefficient

                intensity_map_3D[i,j,k] = voxel_intensity

                if voxel_intensity > max_intensity:
                    max_intensity = voxel_intensity

    intensity_map_3D = np.divide(intensity_map_3D,max_intensity)

    return intensity_map_3D

# courtesy of matplotlib documentation
def explode(data):
    size = np.array(data.shape)*2

    if len(size) == 4:
        size[3] = 4

    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

def plot_virt_image(intensity_map_3D):

    data_shape = intensity_map_3D.shape

    filled = intensity_map_3D > 0.1  
    facecolors = np.zeros(data_shape + (3,))

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for k in range(data_shape[2]):
                facecolors[i,j,k] = intensity_map_3D[i,j,k] * np.ones(3)

    filled_2 = explode(filled)
    facecolors_2 = explode(facecolors)

    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    
    x[0::2, :, :] += 0.3
    y[:, 0::2, :] += 0.3
    z[:, :, 0::2] += 0.3
    x[1::2, :, :] += 0.7
    y[:, 1::2, :] += 0.7
    z[:, :, 1::2] += 0.7

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(
        x, y, z, 
        filled_2, 
        facecolors=1-facecolors_2
    )
    ax.set_aspect('equal')

    plt.show()

image_dimensions = [50,50,50]
centers = get_centers(image_dimensions)

intensity_map_3D = np.zeros(image_dimensions)
for i in range(0,4):
    intensity_map_3D = intensity_map_3D + get_dot_intensities(image_dimensions, centers[i])
intensity_map_3D[intensity_map_3D>1] = 1

plot_virt_image(intensity_map_3D)



