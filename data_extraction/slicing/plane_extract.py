import os

import numpy as np
import pyvista as pv
import vtkmodules.all as vtk
from scipy.spatial import distance
from vtk.util.numpy_support import vtk_to_numpy

import maths_utils


def save_plane_img(cfg, transformed_slice, save_loc):
    """Saves the extracted plane image.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        transformed_slice (pyvista.core.pointset.UnstructuredGrid): Extracted slice that is transformed to xy plane.
        save_loc (str): Location of where to save image to.

    Returns:

    """
    transformed_slice = pv.wrap(transformed_slice)

    transformed_slice.plot(cpos='xy',
                           off_screen=True,
                           show_axes=False,
                           window_size=cfg.DATA_OUT.SAVE_IMG_RESOLUTION,
                           background=cfg.DATA_OUT.SAVE_BCKGD_CLR,
                           anti_aliasing=False,
                           screenshot=save_loc,
                           scalars=cfg.LABELS.LABEL_NAME,
                           show_scalar_bar=False)
    return


def subsample_mesh(cfg, in_mesh):
    """Subsamples the mesh to a lower resolution given a subsampling factor. The subsampled meshes are used as a method
    to calculate centre of masses more quickly than the very high resolution full meshes.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        in_mesh (pyvista.core.pointset.UnstructuredGrid): Entire full resolution cardiac mesh.

    Returns:
        pyvista.core.pointset.UnstructuredGrid: Lower resolution subsampled mesh.
    """
    return in_mesh.extract_cells(range(0, in_mesh.n_cells, cfg.PARAMETERS.SUBSAMPLE_FACTOR))


def prepare_meshes(cfg, mesh_path):
    """Prepares the full cardiac mesh and subsampled mesh for further processing. This entails 1) Loading mesh file
    2) Removing the Aorta 3) Translating mesh to origin and subsampling.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        mesh_path (str): Location of binary mesh file to load from.

    Returns:
        Tuple[pyvista.core.pointset.UnstructuredGrid, pyvista.core.pointset.UnstructuredGrid]: Translated full
        resolution mesh and subsampled mesh.
    """
    case_mesh = pv.get_reader(mesh_path).read()
    case_mesh = pv.wrap(case_mesh).threshold((cfg.LABELS.AORTA - 1, cfg.LABELS.AORTA + 1),
                                             invert=True,
                                             scalars=cfg.LABELS.LABEL_NAME,
                                             preference='cell')

    origin_centred_mesh = maths_utils.translate_mesh_to_origin(case_mesh)
    low_res_mesh = subsample_mesh(cfg, origin_centred_mesh)
    return origin_centred_mesh, low_res_mesh


def slice_with_plane(in_mesh, origin, normal):
    """ Slices a surface mesh with a given origin and normal that defines a slicing plane.

    Args:
        in_mesh (obj): Mesh for slicing.
        origin (list): Origin of plane to perform slicing.
        normal (list): Normal of plane to perform slicing.

    Returns:
        cutter.GetOutput(): Extracted slice object

    """
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(in_mesh)
    cutter.Update()
    return cutter.GetOutput()


def find_lv_apex(cfg, subsampled_mesh):
    """Function to find an intial rough point for the left ventricular apex.

    Args:
        cfg (easydict.EasyDict): configuration file.
        subsampled_mesh (pyvista.core.pointset.UnstructuredGrid): Low resolution mesh to find initial LV apex from.

    Returns:
        Tuple[pyvista.core.pyvista_ndarray.pyvista_ndarray, numpy.ndarray]: Initial LV apex coordinates and valid points
        found in the left ventricle for performing the more computationally expensive ray trace LV apex search
    """
    mitral_valve_centroid = calc_label_com(cfg, subsampled_mesh, cfg.LABELS.MITRAL_VALVE)
    lv_points = pv.wrap(cell_threshold(cfg,
                                       subsampled_mesh,
                                       start=cfg.LABELS.LV,
                                       end=cfg.LABELS.LV)).points

    norm_dist = distance.cdist(np.atleast_2d(mitral_valve_centroid), lv_points, 'euclidean').T
    max_indice = np.argmax(norm_dist)

    apex_coord = lv_points[max_indice, :]
    lv_length = np.linalg.norm([apex_coord, mitral_valve_centroid])
    thresholded_points = []
    for ii in range(len(norm_dist)):
        if norm_dist[ii] > lv_length * cfg.PARAMETERS.THRESHOLD_PERCENTAGE:
            thresholded_points += [lv_points[ii, :]]

    return apex_coord, np.squeeze(thresholded_points)


def find_lv_apex_raytrace(cfg, in_mesh, subsampled_mesh):
    """More accurate method for finding the left ventricular apex using a ray tracing method. An initial fast but
    inaccurate LV apex location is found and from this point the left ventricle is thresholded by a defined percentage
    to avoid wasted computation time, as this is an expensive task. Thresholding also adds robustness as it reduces
    the possibility of finding thin points near base of the left ventricle.

    Args:
        cfg (easydict.EasyDict): configuration file.
        in_mesh (pyvista.core.pointset.UnstructuredGrid): Entire cardiac mesh to find LV apex from.
        subsampled_mesh (pyvista.core.pointset.UnstructuredGrid): Low resolution mesh to find initial LV apex from.

    Returns:
        lv_apex_coords (numpy.ndarray): Accurate coordinates of the left ventricular apex.
    """
    lv_mesh = pv.wrap(in_mesh).threshold((cfg.LABELS.LV, cfg.LABELS.LV),
                                         invert=True,
                                         scalars=cfg.LABELS.LABEL_NAME,
                                         preference='cell')

    mitral_valve_centroid = calc_label_com(cfg, subsampled_mesh, cfg.LABELS.MITRAL_VALVE)

    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(in_mesh.extract_surface())
    obb_tree.BuildLocator()
    points_intersection = vtk.vtkPoints()

    __, threshold_points = find_lv_apex(cfg, subsampled_mesh)  # find fast but inaccurate lv location as starting point

    points_of_intersection = []
    for ii in range(len(threshold_points)):
        code = obb_tree.IntersectWithLine(mitral_valve_centroid,
                                          threshold_points[ii],
                                          points_intersection,
                                          None)  # looks like it isn't called but it is important and must be.

        points_vtk_intersection_data = points_intersection.GetData()
        num_points_intersection = points_vtk_intersection_data.GetNumberOfTuples()

        if num_points_intersection == 2:  # this means the ray has gone through both endo and epicardium
            for idx in range(num_points_intersection):
                _tup0 = points_vtk_intersection_data.GetTuple3(0)
                _tup1 = points_vtk_intersection_data.GetTuple3(1)
                points_of_intersection.append([_tup0, _tup1])

    points_of_intersection = np.squeeze(points_of_intersection)
    thinnest_point = np.argmax(
        np.linalg.norm([points_of_intersection[:, 0, :] - points_of_intersection[:, 1, :]], axis=2))
    lv_apex_coords = points_of_intersection[thinnest_point, 1, :]
    return lv_apex_coords


def calc_label_com(cfg, in_mesh, label):
    """Calculates the centre of mass of a specific label within a mesh containing multiple labels.
    Args:
        cfg (easydict.EasyDict): Configuration file.
        in_mesh (pyvista.core.pointset.UnstructuredGrid): Entire cardiac mesh.
        label (int or list): Value of the label within the mesh.

    Returns:
        numpy.ndarray: Centre of mass of selected label.
    """
    if isinstance(label, int):
        return pv.wrap(cell_threshold(cfg, in_mesh, start=label, end=label)).center_of_mass()
    else:
        return [pv.wrap(cell_threshold(cfg, in_mesh, start=label_val, end=label_val)).center_of_mass() for label_val in
                label]


def fancy_plot(cfg, transformed_slice, transformed_mesh):
    """ Function to visualize how each standard view actually slices through the heart in 3D.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        transformed_slice (pyvista.core.pointset.UnstructuredGrid): Extracted slice that is transformed to xy plane.
        transformed_mesh (pyvista.core.pointset.UnstructuredGrid): Full mesh that is transformed for visualization.
    """
    bounds = list(transformed_slice.bounds)
    bounds[0] = -90
    bounds[1] = 90
    bounds[2] = -90
    bounds[3] = 90

    plane_translation_delta = np.mean(transformed_slice.points, axis=0)[2] - 0.1  # slight offset for visualization
    true_plane = pv.Plane(center=(0, 0, plane_translation_delta),
                          direction=(0, 0, 1),
                          i_size=(bounds[1] - bounds[0]),
                          j_size=bounds[3] - bounds[2])

    plotter = pv.Plotter()
    plotter.add_mesh(transformed_slice,  # Extracted slice
                     scalars=cfg.LABELS.LABEL_NAME,
                     smooth_shading=True,
                     show_scalar_bar=False)
    plotter.add_mesh(true_plane,  # Blank plane
                     color=cfg.DATA_OUT.SAVE_BCKGD_CLR,
                     smooth_shading=True)
    plotter.add_mesh(transformed_mesh,  # Cardiac mesh
                     style='wireframe',
                     color='white',
                     smooth_shading=True,
                     opacity=0.15)
    plotter.show()
    return


def cell_threshold(cfg, in_mesh, start, end):
    """Perform thresholding on individual cell values of a mesh.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        in_mesh (pyvista.core.pointset.UnstructuredGrid): Entire cardiac mesh.
        start (int): Starting value to perform thresholding from.
        end (int): Ending value to perform thresholding to.

    Returns:
        vtkmodules.vtkCommonDataModel.vtkPolyData: Cell thresholded mesh.
    """
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(in_mesh)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, cfg.LABELS.LABEL_NAME)
    threshold.ThresholdBetween(start, end)
    threshold.Update()
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(threshold.GetOutputPort())
    surfer.Update()
    return surfer.GetOutput()


def vtk_cell_field_to_numpy(vtk_variable, array_name):
    return vtk_to_numpy(vtk_variable.GetCellData().GetArray(array_name))


def get_2ch(cfg, subsampled_mesh, lv_apex):
    """Calculates Apical 2 Chamber view coordinates for slicing."""
    rv_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.RV)
    mv_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.MITRAL_VALVE)

    __, out_pnts = maths_utils.pnt2line(tuple(rv_points), tuple(lv_apex), tuple(mv_points))

    out_vec = np.squeeze(rv_points - np.array(out_pnts))

    return out_vec, np.squeeze(out_pnts)


def get_rv_inflow(cfg, subsampled_mesh, right_atrium_mesh_pnts):
    """Calculates Right ventricle inflow view coordinates for slicing."""
    rv_pv_pnts = (calc_label_com(cfg, subsampled_mesh, label=[cfg.LABELS.RV, cfg.LABELS.PULMONARY_VALVE]))
    rv_inflow_pnts = np.vstack((rv_pv_pnts, right_atrium_mesh_pnts))
    return maths_utils.plane_equation_calc(rv_inflow_pnts), np.mean(rv_inflow_pnts, axis=0)


def get_lv_plax(cfg, subsampled_mesh, aortic_valve_mesh_pnts):
    """Calculates left ventricle parasternal long axis view coordinates for slicing."""
    lv_plax_pnts = np.vstack(
        (calc_label_com(cfg, subsampled_mesh, label=[cfg.LABELS.LV, cfg.LABELS.MITRAL_VALVE]),
         aortic_valve_mesh_pnts))
    return maths_utils.plane_equation_calc(lv_plax_pnts), np.mean(lv_plax_pnts, axis=0)


def get_psax_aortic(left_atrium_mesh_pnts, right_atrium_mesh_pnts, aortic_valve_mesh_pnts):
    """Calculates aortic valve level parasternal short axis view coordinates for slicing."""
    psax_aortic_pnts = np.vstack((left_atrium_mesh_pnts, right_atrium_mesh_pnts, aortic_valve_mesh_pnts))
    return maths_utils.plane_equation_calc(psax_aortic_pnts), np.mean(psax_aortic_pnts, axis=0)


def get_psax_mv(vert_vec, heart_com, lv_apex):
    """Calculates mitral valve level parasternal short axis view coordinates for slicing."""
    psax_mv_normal = maths_utils.find_plane_from_normal(vert_vec, -0.25 * (heart_com + lv_apex))
    return tuple([psax_mv_normal, np.mean(np.squeeze(maths_utils.find_points_on_plane(psax_mv_normal)), axis=0)])


def get_psax_pm(vert_vec, heart_com, lv_apex):
    """Calculates papillary muscle level parasternal short axis view coordinates for slicing."""
    psax_pm_normal = maths_utils.find_plane_from_normal(vert_vec, -0.75 * (heart_com + lv_apex))
    return tuple([psax_pm_normal, np.mean(np.squeeze(maths_utils.find_points_on_plane(psax_pm_normal)), axis=0)])


def get_psax_lower(vert_vec, heart_com, lv_apex):
    """Calculates lower level parasternal short axis view coordinates for slicing."""
    psax_lower_normal = maths_utils.find_plane_from_normal(vert_vec, -0.85 * (heart_com + lv_apex))
    return tuple([psax_lower_normal, np.mean(np.squeeze(maths_utils.find_points_on_plane(psax_lower_normal)), axis=0)])


def get_a4c(left_atrium_mesh_pnts, right_atrium_mesh_pnts, lv_apex):
    """Calculates Apical 4 Chamber view coordinates for slicing."""
    a4c_pnts = np.vstack((left_atrium_mesh_pnts, right_atrium_mesh_pnts, lv_apex))
    return maths_utils.plane_equation_calc(a4c_pnts), np.mean(a4c_pnts, axis=0)


def get_a5c(left_atrium_mesh_pnts, aortic_valve_mesh_pnts, lv_apex):
    """Calculates Apical 5 Chamber view coordinates for slicing."""
    a5c_pnts = np.vstack((left_atrium_mesh_pnts, aortic_valve_mesh_pnts, lv_apex))
    return maths_utils.plane_equation_calc(a5c_pnts), np.mean(a5c_pnts, axis=0)


def get_cardiac_images(cfg, in_mesh, subsampled_mesh):
    """Function to calculate commonly used landmarks for slicing and calling functions for all the selected views.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        in_mesh (pyvista.core.pointset.UnstructuredGrid): Entire full resolution cardiac mesh.
        subsampled_mesh (pyvista.core.pointset.UnstructuredGrid):  Low resolution mesh.

    Returns:
        returned_points4imgs (list): coordinate information for all the selected views.
    """
    heart_com = np.array(in_mesh.center)
    lv_apex = find_lv_apex_raytrace(cfg, in_mesh, subsampled_mesh)
    vert_vec = (lv_apex - heart_com) / np.linalg.norm(lv_apex - heart_com)

    l_atrium_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.LA)
    r_atrium_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.RA)
    aortic_valve_points = calc_label_com(cfg, subsampled_mesh, label=cfg.LABELS.AORTIC_VALVE)

    returned_points4imgs = []
    if 'rv_inflow' in cfg.DATA_OUT.SELECTED_VIEWS:
        rv_inflow = get_rv_inflow(cfg, subsampled_mesh, r_atrium_points)
        returned_points4imgs += [rv_inflow]
    if 'lv_plax' in cfg.DATA_OUT.SELECTED_VIEWS:
        lv_plax = get_lv_plax(cfg, subsampled_mesh, aortic_valve_points)
        returned_points4imgs += [lv_plax]
    if 'psax_aortic' in cfg.DATA_OUT.SELECTED_VIEWS:
        psax_aortic = get_psax_aortic(l_atrium_points, r_atrium_points, aortic_valve_points)
        returned_points4imgs += [psax_aortic]
    if 'psax_mv' in cfg.DATA_OUT.SELECTED_VIEWS:
        psax_mv = get_psax_mv(vert_vec, heart_com, lv_apex)
        returned_points4imgs += [psax_mv]
    if 'psax_pm' in cfg.DATA_OUT.SELECTED_VIEWS:
        psax_pm = get_psax_pm(vert_vec, heart_com, lv_apex)
        returned_points4imgs += [psax_pm]
    if 'psax_lower' in cfg.DATA_OUT.SELECTED_VIEWS:
        psax_lower = get_psax_lower(vert_vec, heart_com, lv_apex)
        returned_points4imgs += [psax_lower]
    if 'a4c' in cfg.DATA_OUT.SELECTED_VIEWS:
        a4c = get_a4c(l_atrium_points, r_atrium_points, lv_apex)
        returned_points4imgs += [a4c]
    if 'a5c' in cfg.DATA_OUT.SELECTED_VIEWS:
        a5c = get_a5c(l_atrium_points, aortic_valve_points, lv_apex)
        returned_points4imgs += [a5c]
    if 'a2c' in cfg.DATA_OUT.SELECTED_VIEWS:
        a2c = get_2ch(cfg, subsampled_mesh, lv_apex)
        returned_points4imgs += [a2c]

    return returned_points4imgs


def run_slice_extraction(cfg, mesh_path, case_save_path, show_fancyplot=False):
    """Main function to call relevant calculation and plotting functions.

    Args:
        cfg (easydict.EasyDict): Configuration file.
        mesh_path (str): Path from which to load binary mesh.
        case_save_path (str): Path to save images to.
        show_fancyplot (bool): Show a plot visualizing how each view plane slices a heart in 3D.
    """
    mesh_origin, subsampled_mesh = prepare_meshes(cfg, mesh_path)

    cardiac_out_points = get_cardiac_images(cfg, mesh_origin, subsampled_mesh)

    for view_index, view_data in enumerate(cardiac_out_points):
        normal = view_data[0]
        plane_origin = view_data[1]

        if len(normal) == 4:
            normal = normal[:3]

        sliced = slice_with_plane(mesh_origin, origin=plane_origin, normal=normal)
        transformed_slice = pv.wrap(maths_utils.rotate_to_xy_plane(sliced, normal))

        if show_fancyplot:
            transformed_mesh = maths_utils.rotate_to_xy_plane(mesh_origin, normal)
            fancy_plot(cfg, transformed_slice, transformed_mesh)
        if case_save_path:
            save_plane_img(cfg, transformed_slice, os.path.join(case_save_path, str(view_index) + '.png'))

    return
