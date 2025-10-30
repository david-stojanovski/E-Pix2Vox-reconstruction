import os
import glob
import time
from itertools import repeat
from multiprocessing import Pool

from config import cfg
from natsort import natsorted
from plane_extract import run_slice_extraction


def main() -> None:
    all_data_paths = natsorted(glob.glob(cfg.DATA_IN.DATA_FOLDER + "*.vtk"))
    time.perf_counter()
    case_names = [file.split(os.sep)[-1].split(".")[0] for file in all_data_paths]

    save_paths = [
        os.path.join(cfg.DATA_IN.SAVE_FOLDER, "heart_seg", "heart_render", "heart", case_name)
        for case_name in case_names
    ]

    for path in save_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    with Pool(cfg.PARAMETERS.NUM_WORKERS) as pool:
        pool.starmap(
            run_slice_extraction,
            zip(repeat(cfg), all_data_paths, save_paths, repeat(cfg.DATA_OUT.FANCY_PLOT)),
        )

    time.perf_counter()


if __name__ == "__main__":
    main()
