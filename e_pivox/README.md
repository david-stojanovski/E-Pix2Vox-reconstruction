# E-Pix2Vox++ &middot; [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://github.com/your/your-project/blob/master/LICENSE)
> Repository containing source code to implement "Efficient Pix2Vox++ for 3D Cardiac Reconstruction from 2D echo views"
> 
> The original Pix2Vox++ networks can be found at: https://gitlab.com/hzxie/Pix2Vox.


## Get Started (3D Cardiac Reconstruction training)
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

## New additions to Pix2Vox++

Ease of use additions have been made to the Pix2Vox++ original code that does not change any fundamental properties.

Within the config file the following additions are shown below:

- Option to specify model size using `__C.NETWORK.MODEL_SIZE`. This can be either `32`, `64` or `128`.
- Boolean flag of whether to use E-PiVox or standard Pix2Vox++ models using `__C.NETWORK.USE_EP2V`.
- Option to specify loss function using `__C.NETWORK.LOSS_FUNC` (case-insensitive). A number of loss functions have 
been implented from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook. The list of
functions can be found inside `e_pixvox/utils/network_losses.py`.
- Option to specify the number of test volumes to save per epoch using `__C.CONST.TEST_SAVE_NUMBER`
- Option to specify save format of test volumes using `__C.TEST.VOL_OR_RENDER_SAVE`. This can be either `volume` which 
saves original sized numpy 3D volumes of each test case (much faster than original option), or the original `render` 
which renders an image of the volume but is very slow to do so.
- Option to specify path to save an Excel file containing individual IoU scores for each test case using
`__C.DIR.IOU_SAVE_PATH`.

Additionally:
- There will be a file within the output folder which prints out all the of parameters used in the `config.py` file 
during the previous training run.
- `.npy` format files can now be used as input for the 3D ground truth models. 
- If the input images are a grayscale image (1 color channel instead of normal 3 for RGB) the color channel will be 
stacked to avoid error.

## Licensing
This project is open sourced under MIT license. (See `LICENSE` for further details)