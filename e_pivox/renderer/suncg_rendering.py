# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-27 16:17:28
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-09-29 03:21:29
# @Email:  cshzxie@gmail.com

import bpy
import bpy_extras
import cv2
import json
import math
import mathutils
import numpy as np
import os
import random
import shutil
import sys


def parent_obj_to_camera(b_camera, origin):
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty    # setup parenting
    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


# =================================================================================================

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

# Input parameters
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
SUNCG_FOLDER = sys.argv[-4]
SHAPENET_FOLDER = sys.argv[-3]
OUTPUT_FOLDER = sys.argv[-2]
scene_id = sys.argv[-1]
# SCRIPT_FOLDER = '/home/hzxie/Development/3D-Reconstruction-Utilties/SuncgRendering'
# scene_id = '2c13f9166818a17032720608b30dabe5'
# SUNCG_FOLDER = '/home/hzxie/Datasets/SUNCG/'
# SHAPENET_FOLDER = '/home/hzxie/Datasets/ShapeNetCore.v1'
# OUTPUT_FOLDER = '/home/hzxie/Datasets/Things1M/'
SIZE_DIFF_LIMIT = 0.5
OCCLUDED_LIMT = 0.125
MAX_TRY_TIMES = 15
OBJECT_FOLDER = os.path.join(SUNCG_FOLDER, 'object')
CATEGORY_MAPPING = os.path.join(SUNCG_FOLDER, 'metadata', 'ModelCategoryMapping.csv')
ROOM_FOLDER = os.path.join(SUNCG_FOLDER, 'room', scene_id)
CONFIG_FILE = os.path.join(SUNCG_FOLDER, 'house', scene_id, 'house.json')
ACCEPTED_CATEGORIES = [
    '03001627', '03211117', '03636649', '03367059', '04256520', '04379243', '03063968', '03211616', '03928116'
]

with open(os.path.join(SCRIPT_FOLDER, 'ShapeNetCore.json')) as f:
    SHAPENET_OBJECTS = json.loads(f.read())

with open(os.path.join(SCRIPT_FOLDER, 'taxonomy.json')) as f:
    PARENT_CATEGORY_MAPPING = {}
    _categories = json.loads(f.read())
    for c in _categories:
        if len(c['children']) == 0:
            continue
        c_id = c['synsetId']
        children = c['children']
        PARENT_CATEGORY_MAPPING[c_id] = c_id
        for ch in children:
            PARENT_CATEGORY_MAPPING[ch] = c_id

# for c in _categories:
#     c_id = c['synsetId']
#     if c_id not in ACCEPTED_CATEGORIES:
#         ACCEPTED_CATEGORIES.append(c_id)

#     children = c['children']
#     for ch in children:
#         if ch not in ACCEPTED_CATEGORIES:
#             ACCEPTED_CATEGORIES.append(ch)

# Import Room Skeletons
skeletons = os.listdir(ROOM_FOLDER)
for sk in skeletons:
    # Ignore ceils
    if sk[-5] == 'c':
        continue
    sk = os.path.join(ROOM_FOLDER, sk)
    bpy.ops.import_scene.obj(filepath=sk)

category_mapping = {}
with open(CATEGORY_MAPPING) as f:
    # Ignore first five lines
    for i in range(6):
        line = f.readline()
    while line:
        line = line.split(',')
        model_id = line[1]
        category = line[6]
        category_mapping[model_id] = category[1:]
        line = f.readline()

with open(CONFIG_FILE) as f:
    cfg = json.loads(f.read())

rooms = []
objects = {}
levels = cfg['levels']
for lvl_idx, lvl in enumerate(levels):
    for node in lvl['nodes']:
        if node['type'] == 'Room':
            if not 'nodeIndices' in node:
                continue
            rooms.append({
                'lvl_idx': lvl_idx,
                'bbox': node['bbox'],
                'node_indices': node['nodeIndices'],
                'model_id': node['modelId'],
                'nodes': []
            })
        else:
            if not node['type'] == 'Object':
                print('Unknown Node Type: %s' % node['type'])
                continue
            objects[node['id']] = {'model_id': node['modelId'], 'transform': node['transform'], 'bbox': node['bbox']}
            # TODO: Handle mirror and scale on rooms

# Set rendering options
scene = bpy.context.scene
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'
#bpy.context.scene.render.use_shadows = False
#bpy.context.scene.render.use_raytrace = False

lamp = bpy.data.lamps['Lamp']
lamp.distance = 15
lamp.type = 'POINT'
lamp.energy = random.random() * 2.25 + 0.75
lamp.use_specular = False

# Make the lamp track the object
sun = bpy.data.objects['Lamp']
lamp_constraint = sun.constraints.new(type='TRACK_TO')
lamp_constraint.track_axis = 'TRACK_NEGATIVE_Z'
lamp_constraint.up_axis = 'UP_Y'

# Make the camera track the object
bpy.data.cameras['Camera'].lens = 96
cam = scene.objects['Camera']
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

# Put objects in rooms
for r_idx, r in enumerate(rooms):
    # Setup the wall and ground
    wall_obj = rooms[r_idx]['model_id']
    # bpy.ops.import_scene.obj(filepath=os.path.join(ROOM_FOLDER, '%sc.obj' % wall_obj))
    # bpy.ops.import_scene.obj(filepath=os.path.join(ROOM_FOLDER, '%sf.obj' % wall_obj))
    # bpy.ops.import_scene.obj(filepath=os.path.join(ROOM_FOLDER, '%sw.obj' % wall_obj))
    # Process all objects in rooms
    for ni in r['node_indices']:
        obj_key = '%d_%d' % (r['lvl_idx'], ni)
        if not obj_key in objects:
            continue
        #
        obj = objects[obj_key]
        obj_name = 'Model%s' % obj_key
        model_id = obj['model_id']
        category = category_mapping[model_id]
        if category in ['03063968', '05217688']:
            continue
        #
        category = PARENT_CATEGORY_MAPPING[category] if category in PARENT_CATEGORY_MAPPING else category
        obj['category'] = category
        # Get the size of the original object
        bpy.ops.import_scene.obj(filepath=os.path.join(OBJECT_FOLDER, model_id, '%s.obj' % model_id))
        bpy.context.selected_objects[0].name = obj_name
        bound_box = np.array([v[:] for v in bpy.data.objects[obj_name].bound_box])
        z_min = np.min(bound_box, axis=0)[1]
        z_max = np.max(bound_box, axis=0)[1]
        size = np.max(bound_box, axis=0) - np.min(bound_box, axis=0)
        # Replace another 3D model in the same category
        if category in SHAPENET_OBJECTS and category not in ['02933112', '03691459', '02958343']:
            bpy.data.objects[obj_name].select = True
            bpy.ops.object.delete()
        #
        scale = 1
        try_times = 0
        is_fitted = False
        fallback_model_id = model_id
        while not is_fitted and try_times < MAX_TRY_TIMES and category in SHAPENET_OBJECTS and category not in [
                '02933112', '03691459'
        ]:
            model_id = random.choice(SHAPENET_OBJECTS[category])
            obj['model_id'] = model_id
            bpy.ops.import_scene.obj(filepath=os.path.join(SHAPENET_FOLDER, category, model_id, 'model.obj'))
            # Merge multiple meshes into one
            _objects = bpy.context.selected_objects
            ctx = bpy.context.copy()
            ctx['active_object'] = _objects[0]
            ctx['selected_objects'] = _objects
            ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in _objects]
            bpy.ops.object.join(ctx)
            bpy.context.selected_objects[0].name = obj_name
            # Get the size of the imported object
            _bound_box = np.array([v[:] for v in bpy.data.objects[obj_name].bound_box])
            _z_min = np.min(_bound_box, axis=0)[1]
            _z_max = np.max(_bound_box, axis=0)[1]
            # Resize the object
            scale = (z_max - z_min) / (_z_max - _z_min)
            # Rotate the object and move above the floor
            z_offset = -z_min
            if scale > 1:
                z_offset = -_z_min * ((scale - 1) / 2 + 1)
            elif scale < 1:
                z_offset = -_z_min * (1 - (1 - scale) / 2)
            # bpy.context.selected_objects[0].data.transform(mathutils.Matrix.Rotation(radians(-90.0), 4, 'Y'))
            bpy.data.objects[obj_name].data.transform(
                mathutils.Matrix(((0, 0, -1, 0), (0, 1, 0, 0.15 + z_offset), (1, 0, 0, 0), (0, 0, 0, 1))))
            bpy.data.objects[obj_name].data.update()
            _bound_box = np.array([v[:] for v in bpy.data.objects[obj_name].bound_box])[::-1]
            _size = (np.max(_bound_box, axis=0) - np.min(_bound_box, axis=0)) * scale
            print(size, _size, _size - size, np.sum((_size - size) <= SIZE_DIFF_LIMIT))
            if np.sum((_size - size) <= SIZE_DIFF_LIMIT) == 3:
                is_fitted = True
            else:
                try_times += 1
                bpy.ops.object.delete()
        # Fallback to the original 3D model
        if not is_fitted and try_times >= MAX_TRY_TIMES:
            bpy.ops.import_scene.obj(filepath=os.path.join(OBJECT_FOLDER, fallback_model_id, '%s.obj' %
                                                           fallback_model_id))
            bpy.context.selected_objects[0].name = obj_name
        # Apply the transform to the object
        transform = obj['transform']
        bpy.data.objects[obj_name].data.transform(
            mathutils.Matrix((
                (transform[0], transform[4], transform[8], transform[12]),    #  - r_centroid_x
                (transform[1], transform[5], transform[9], transform[13]),    #  - r_centroid_z
                (transform[2], transform[6], transform[10], transform[14]),    #  - r_centroid_y
                (transform[3], transform[7], transform[11], transform[15]))))
        bpy.data.objects[obj_name].data.update()
        # Resize the object
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        bpy.data.objects[obj_name].scale = (scale, scale, scale)

bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
# Rendering for objects in rooms
for r_idx, r in enumerate(rooms):
    for ni in r['node_indices']:
        obj_key = '%d_%d' % (r['lvl_idx'], ni)
        if not obj_key in objects:
            continue
        #
        obj = objects[obj_key]
        obj_name = 'Model%s' % obj_key
        model_id = obj['model_id']
        if 'category' not in obj or len(model_id) < 8:
            continue
        #
        category = obj['category']
        # Skip not accepted categories
        print(category, obj_name, model_id)
        if category not in ACCEPTED_CATEGORIES or obj_name not in bpy.data.objects:
            continue
        # Get object center
        obj_center = (np.array(obj['bbox']['max']) + np.array(obj['bbox']['min'])) / 2
        # Set up light
        sun_empty = parent_obj_to_camera(sun, (obj_center[0], -obj_center[2], obj_center[1]))
        lamp_constraint.target = sun_empty
        # Set up camera
        cam_empty = parent_obj_to_camera(cam, (obj_center[0], -obj_center[2], obj_center[1]))
        cam_constraint.target = cam_empty
        # Set up output folder
        output_folder = os.path.join(OUTPUT_FOLDER, category, model_id, scene_id)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        img_counter = 0
        # Set up camera and light position
        for az in range(0, 360, 15):
            rotation_euler_x = math.radians(60)
            rotation_euler_y = math.radians(0)
            rotation_euler_z = math.radians(az)
            rotation_euler = (rotation_euler_x, rotation_euler_y, rotation_euler_z)
            sun_empty.rotation_euler = rotation_euler
            cam_empty.rotation_euler = rotation_euler
            cam.location = sun.location
            # Switch to global view and render
            final_file_path = '/tmp/%s_%s_%s_final.png' % (model_id, scene_id, az)
            scene.render.filepath = final_file_path
            bpy.ops.render.render(write_still=True)
            # Switch to local view and render
            ## Set all objects to invisible
            for _object_name, _object in bpy.data.objects.items():
                _object.hide_render = True
            ## Set current object to visible
            bpy.data.objects[obj_name].hide_render = False
            sun.hide_render = False
            ## Render the current object
            clean_file_path = '/tmp/%s_%s_%s_clean.png' % (model_id, scene_id, az)
            scene.render.filepath = clean_file_path
            bpy.ops.render.render(write_still=True)
            # Set all object to visible
            for _object_name, _object in bpy.data.objects.items():
                _object.hide_render = False
            # Check if current view is occluded
            clean_img = cv2.imread(clean_file_path, -1)
            final_img = cv2.imread(final_file_path, -1)
            alpha = clean_img[:, :, 3]
            mask = (alpha == 255).astype(np.uint8)
            clean_img = clean_img[:, :, :3]
            final_img = final_img[:, :, :3]
            mask = mask[:, :, np.newaxis]
            error_map = (final_img * mask != clean_img).astype(np.float32)
            kernel = np.ones((3, 3), np.uint8)
            error_map = cv2.dilate((1 - error_map), kernel, iterations=1)
            error_map = 1 - error_map
            _occluded = np.sum(error_map / (np.sum(mask) * 3))
            if _occluded <= OCCLUDED_LIMT:
                shutil.move(clean_file_path, os.path.join(output_folder, 'render_%02d_clean.png' % img_counter))
                shutil.move(final_file_path, os.path.join(output_folder, 'render_%02d_final.png' % img_counter))
                img_counter += 1
            else:
                # shutil.move(clean_file_path , '/tmp/blender/%s_%s_%s_%s_%.3f_clean.png' % (category, model_id, scene_id, az, _occluded))
                # shutil.move(final_file_path , '/tmp/blender/%s_%s_%s_%s_%.3f_final.png' % (category, model_id, scene_id, az, _occluded))
                os.remove(clean_file_path)
                os.remove(final_file_path)
        # Remove empty folder
        if len(os.listdir(output_folder)) == 0:
            os.rmdir(output_folder)
        parent_dir = os.path.abspath(os.path.join(output_folder, os.pardir))
        if len(os.listdir(parent_dir)) == 0:
            os.rmdir(parent_dir)
        # Remove constraints
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[sun_empty.name].select = True
        bpy.data.objects[cam_empty.name].select = True
        bpy.ops.object.delete()
