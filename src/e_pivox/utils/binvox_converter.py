import os
import sys
import subprocess
from glob import glob
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import binvox_rw
from loguru import logger


def main() -> None:
    if len(sys.argv) != 2:
        logger.error("python binvox_converter.py input_file_folder")
        sys.exit(1)

    input_file_folder = sys.argv[1]
    if not Path(input_file_folder).exists() or not Path(input_file_folder).is_dir():
        logger.error("Input folder not exists!")
        sys.exit(2)

    n_vox = 32
    mesh_extension = "*.off"

    folder_path = Path(input_file_folder) / mesh_extension
    mesh_files = glob(str(folder_path))

    for m_file in mesh_files:
        file_path = Path(input_file_folder) / m_file
        file_name, _ = os.path.splitext(m_file)
        binvox_file_path = Path(input_file_folder) / f"{file_name}.binvox"

        if binvox_file_path.exists():
            logger.warning(f"{dt.now()} File: {binvox_file_path} exists. It will be overwritten.")
            binvox_file_path.unlink()

        logger.info(f"[INFO] {dt.now()} Processing file: {file_path}")
        rc = subprocess.call(
            [
                "binvox",
                "-d",
                str(n_vox),
                "-e",
                "-cb",
                "-rotx",
                "-rotx",
                "-rotx",
                "-rotz",
                m_file,
            ]
        )
        if rc != 0:
            logger.warning(f"[WARN] {dt.now()} Failed to convert file: {m_file}")
            continue

        with binvox_file_path.open("rb") as file:
            v = binvox_rw.read_as_3d_array(file)

        v.data = np.transpose(v.data, (2, 0, 1))
        with binvox_file_path.open("wb") as file:
            binvox_rw.write(v, file)


if __name__ == "__main__":
    return_code = subprocess.call(["which", "binvox"], stdout=subprocess.PIPE)
    if return_code == 0:
        main()
    else:
        logger.error(f"[FATAL] {dt.now()} Please make sure you have binvox installed.")
