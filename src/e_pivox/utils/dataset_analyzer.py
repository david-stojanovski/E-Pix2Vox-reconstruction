import sys
from queue import Queue
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import imageio
from config import cfg
from loguru import logger


def main() -> None:
    input_file_folder = Path(cfg.DATASETS.HEARTSEG.IMG_ROOT)
    if not input_file_folder or not input_file_folder.is_dir():
        logger.error("Input folder not exists!")
        sys.exit(2)

    file_name_pattern = "*.png"
    folders_to_explore = Queue()
    folders_to_explore.put(input_file_folder)

    total_files = 0
    mean = np.asarray([0.0, 0.0, 0.0])
    std = np.asarray([0.0, 0.0, 0.0])
    while not folders_to_explore.empty():
        current_folder = folders_to_explore.get()

        if not current_folder or not current_folder.is_dir():
            logger.warning(f"Ignore folder: {current_folder}")
            continue

        logger.info(f"Listing files in folder: {current_folder}")
        n_folders = 0
        n_files = 0
        files = current_folder.iterdir()
        for file_name in files:
            file_path = current_folder / file_name
            if file_path.is_dir():
                n_folders += 1
                folders_to_explore.put(file_path)
            elif file_path.is_file() and fnmatch(file_name, file_name_pattern):
                n_files += 1
                total_files += 1

                img = imageio.imread(file_path)
                img_mean = np.mean(img, axis=(0, 1))
                img_std = np.var(img, axis=(0, 1))
                mean += img_mean
                std += img_std
    logger.info(f"Mean = {mean / total_files}, Std = {np.sqrt(std) / total_files}")


if __name__ == "__main__":
    main()
