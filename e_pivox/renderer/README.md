# Things3D Rendering Instruction

## Download

You can obtain the rendering images of the Things3D dataset [here](https://gateway.infinitescript.com/?fileName=Things3D).

## Prerequisites

- Blender < 2.80 (Tested on Blender 2.79)
- [ShapeNetCore.v1](http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip)
- [SUNCG](https://sscnet.cs.princeton.edu/)

Due to the [lawsuit](https://futurism.com/tech-suing-facebook-princeton-data) between Planner 5D and Princeton, SUNCG is no longer available. Moreover, according to the regulation of [Planner 5D](https://planner5d.com/), anyone cannot redistribute the SUNCG dataset.

Although we cannot provide you the SUNCG dataset, you can still apply this rendering procedure to other similar datasets such as [3D-FRONT](https://pages.tmall.com/wow/cab/tianchi/promotion/alibaba-3d-scene-dataset) and [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future).

## Rendering

First, you need to specify the values of following variables.

```
SUNCG_ROOT=/path/to/SUNCG
SHAPENET_FOLDER=/path/to/ShapeNet.Core.v1
OUTPUT_FOLDER=/path/to/output
HOUSE_ID="2c13f9166818a17032720608b30dabe5"
```

Then, use the following commands to generate the renderings of the house.

```bash
BLENDER=/path/to/blender

$BLENDER --background --python suncg_rendering.py -- $SUNCG_ROOT $SHAPENET_FOLDER $OUTPUT_FOLDER $HOUSE_ID
```

