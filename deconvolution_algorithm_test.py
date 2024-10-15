import numpy as np

virt_image = np.zeros((1000,1000,5000))

rng = np.random.default_rng()

# create 125 points to place in sample 
for i in range(0,5):

    x_region_depth = i*200
    x_reals = rng.poisson(x_region_depth+100,25)

    for j in range(0,5):

        y_region_depth = j*200
        y_reals = rng.poisson(y_region_depth+100,5)

        for k in range(0,5):
             
            z_region_depth = k*1000
            z_reals = rng.poisson(z_region_depth+500,1)

            center_of_voxel = [
                x_reals[j*5+k], 
                y_reals[k], 
                z_reals, 
            ]

            print(center_of_voxel)



def get_dot_intensities(x_center,y_center,z_center):
    1


    



