import glob
import os

from natsort import natsorted

from config import cfg
from plane_extract import run_slice_extraction


def main():
    # Change these values
    data_folder = '/folder/containing/binary/meshes/'
    save_folder = r'/folder/to/save/slices/to'
    #

    all_data_paths = natsorted(glob.glob(data_folder + '*.vtk'))
    for file in all_data_paths:
        case_name = file.split(os.sep)[-1].split('.')[0]
        save_path = os.path.join(save_folder, 'heart_seg', 'heart_render', 'heart', case_name)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        run_slice_extraction(cfg, file, save_path, show_fancyplot=False)
        print('Finished file:', file)


if __name__ == '__main__':
    main()
