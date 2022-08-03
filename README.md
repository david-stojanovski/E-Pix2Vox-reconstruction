# E-Pix2Vox++ &middot; [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://github.com/your/your-project/blob/master/LICENSE) [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://doi.org/10.48550/arxiv.2207.13424)
> Repository containing source code to implement "Efficient Pix2Vox++ for 3D Cardiac Reconstruction from 2D echo views"


## Cite this work

```
@misc{https://doi.org/10.48550/arxiv.2207.13424,
  doi = {10.48550/ARXIV.2207.13424},
  url = {https://arxiv.org/abs/2207.13424},
  author = {Stojanovski, David and Hermida, Uxio and Muffoletto, Marica and Lamata, Pablo and Beqiri, Arian and Gomez, Alberto},
  keywords = {Image and Video Processing (eess.IV), Artificial Intelligence (cs.AI), Computer Vision and Pattern Recognition (cs.CV), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Efficient Pix2Vox++ for 3D Cardiac Reconstruction from 2D echo views},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Datasets

We use the [ShapeNet](https://www.shapenet.org/) and "Virtual cohort of 1000 synthetic heart meshes from adult human healthy population" datasets in our experiments, which are available below:

- ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- 1000 synthetic heart meshes: https://zenodo.org/record/4506930#.YuEntdLMJ1M



## Installing 
How to set up the appropriate environment to run the code:

```shell
git clone https://github.com/david-stojanovski/E-Pix2Vox-reconstruction.git
cd E-Pix2Vox-reconstruction/
pip install -r requirements.txt
```


## Get Started (Data Generation)
For the cardiac mesh data set, after it has been downloaded and all files placed in a single folder, the following 
should be changed:

- In `data_extraction/convert_ascii2binary.py`:
```python
data_folder = r'/path/to/your/vtkdata/'
save_folder = r'/path/to/your/data_save_folder'
```

- In `data_extraction/run_plane_extract.py`:
```python
data_folder = '/folder/containing/binary/meshes/'
save_folder = r'/folder/to/save/slices/to'
```

- In `data_extraction/voxelize_pyntcloud.py`:
```python
data_folder = r'/path/to/heart/meshes/'
save_folder = r'/path/to/save/voxel_models/to'
```

After this you can run:
```shell
python convert_ascii2binary.py
python run_plane_extract.py
python voxelize_pyntcloud.py
```

You should now have all the primary data required to perform 3D reconstruction using the deep learning networks. 
These folders should now be moved over to the "/e_pivox/datasets" folder.

## Get Started (3D Cardiac Reconstruction)
You must at least update the following in the `config.py` file:

```python
__C.DATASETS.HEARTSEG.IMG_ROOT = '/home/e_pivox/datasets/heart_seg/heart_render'
__C.DATASETS.HEARTSEG.TAXONOMY_FILE_PATH = '/home/e_pivox/datasets/HeartSeg.json'
__C.DATASETS.HEARTSEG.VOXEL_PATH = '/home/e_pivox/datasets/heart_seg/voxel_volumes/%s/%s/model.npy'

__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = r'/home/e_pivox/datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH = r'/home/e_pivox/datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = r'/home/e_pivox/datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
```

To train the networks, you can then run:

```shell
python runner.py
```

## Licensing
This project is open sourced under MIT license. (See `LICENSE` for further details)
