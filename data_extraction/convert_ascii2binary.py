import glob
import os

import pyvista as pv
from natsort import natsorted


def main():
    # Change these values
    data_folder = r'/path/to/your/vtkdata/'
    save_folder = r'/path/to/your/data_save_folder'
    #

    all_data_paths = natsorted(glob.glob(data_folder + '*.vtk'))

    for file in all_data_paths:
        case_mesh = pv.get_reader(file).read()
        case_number = file.split(os.sep)[-1].split('.')[0].split('_')[-1]
        filled_case_name = 'full_heart_mesh_' + str(int(case_number)).zfill(3)
        save_path = os.path.join(save_folder, 'heart_seg', 'heart_render', 'heart', filled_case_name)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        case_mesh.save(os.path.join(save_path, filled_case_name + '.vtk'), binary=True)
        print('Finished file:', file)


if __name__ == '__main__':
    main()
