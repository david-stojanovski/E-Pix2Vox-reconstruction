import glob
import os

import numpy as np
import pandas as pd
from natsort import natsorted
from pyntcloud import PyntCloud

from data_extraction.slicing.config import cfg
from data_extraction.slicing.plane_extract import prepare_meshes


def convert_vtk2np_voxels(in_mesh, vol_resolution):
    df = pd.DataFrame(data=in_mesh.points, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)

    voxelgrid_id = cloud.add_structure('voxelgrid', n_x=vol_resolution, n_y=vol_resolution, n_z=vol_resolution)
    voxelgrid = cloud.structures[voxelgrid_id]
    binary_voxel_array = voxelgrid.get_feature_vector(mode='binary')
    # binary_voxel_array = closing(Binary_voxel_array, cube(2))
    return binary_voxel_array


def main():
    # Change these values
    data_folder = r'/path/to/heart/meshes/'
    save_folder = r'/path/to/save/voxel_models/to'
    vol_resolution = 64
    #

    all_data_paths = natsorted(glob.glob(os.path.join(data_folder, '*.vtk')))

    for file in all_data_paths:
        loaded_mesh, __ = prepare_meshes(cfg, file)
        out_voxel_array = convert_vtk2np_voxels(loaded_mesh, vol_resolution)

        case_name = file.split(os.sep)[-1].split('.')[0]
        save_path = os.path.join(save_folder, 'heart', case_name)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        np.save(os.path.join(save_path, 'model.npy'), out_voxel_array)
        print('Finished file:', file)


if __name__ == '__main__':
    main()
