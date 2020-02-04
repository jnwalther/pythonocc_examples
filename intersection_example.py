from __future__ import print_function

import logging
import numpy as np
import timeit

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.Geom import Geom_Line, Geom_TrimmedCurve
from OCC.Core.GeomConvert import geomconvert_SurfaceToBSplineSurface
from OCC.Core.GeomAPI import GeomAPI_ExtremaCurveSurface
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.TopoDS import TopoDS_Shape, topods_Face
from OCC.Display.SimpleGui import init_display

logging.getLogger('matplotlib').setLevel(logging.ERROR)


def trimmed_curve_from_points(points=[[0.9, 0.9, 0.9]], vecs=[[1, 1, 0]]):
    """
    Create line objects from points/vectors
    """
    return [Geom_TrimmedCurve(Geom_Line(gp_Pnt(*p), gp_Dir(*vec)), 0., 8.)
            for p, vec in zip(points, vecs)]


def show_objects(objs=[], wireframe=False):
    """
    Display pythonOCC geometry objects in simple viewer

    Parameters
    ----------
    objs : list
        List of pythonOCC objects to be displayed
    wireframe : bool
        Set to true to plot wireframe instead of surface
    """
    display, start_display, add_menu, add_function_to_menu = init_display(size=(1920, 1080))
    display.EraseAll()
    if wireframe:
        display.SetModeWireFrame()
    for obj in objs:
        display.DisplayShape(obj, update=True)
    start_display()


def read_brep(filename):
    """
    Retrieve BSpline surface from brep file
    """
    output_shape = TopoDS_Shape()
    builder = BRep_Builder()
    breptools_Read(output_shape, filename, builder)
    brep_face = BRep_Tool.Surface(topods_Face(output_shape))
    return geomconvert_SurfaceToBSplineSurface(brep_face)


def generate_intersection_vectors(c, nxu=100, nxv=10):
    """
    Generate the intersection vectors
    """
    x = np.linspace(0.01*c, 1.99*c, nxu)
    vec_phi = np.linspace(0, 2*np.pi, nxv)
    points = np.pad(x[:, None], ((0, 0), (0, 2)), mode='constant')
    vecs = np.column_stack([np.zeros(nxv), np.cos(vec_phi), np.sin(vec_phi)])
    points, vecs = np.broadcast_arrays(points[:, None, :], vecs[None, ...])
    return points.reshape(-1, 3), vecs.reshape(-1, 3)


def intersect(occsurf, occvecs, v, show_occ=False):
    '''
    Perform intersection
    '''
    occpts = []
    xpts = []
    fails = []
    times = []
    for curve, v_1, v_2 in zip(occvecs, *v):
        failure = [np.nan, np.nan, np.nan]
        t = timeit.timeit(lambda: GeomAPI_ExtremaCurveSurface(curve, occsurf, 0., 1., 0., 1., 0., 1.),
                          number=1)
        x = GeomAPI_ExtremaCurveSurface(curve, occsurf, 0., 6., 0., 1., 0., 1.)
        w, _, _ = x.LowerDistanceParameters()
        times.append(t)
        vec = curve.Value(w)
        if x.LowerDistance() > 1e-4:
            failure = np.array([vec.X(), vec.Y(), vec.Z()])
        occpts.append(vec)
        xpts.append(np.array([vec.X(), vec.Y(), vec.Z()]))
        fails.append(failure)

    if show_occ:
        show_objects([occsurf] + occvecs + occpts)

    print('----------')
    print('Time for intersections', sum(times))

    xpts = np.array(xpts)
    fails = np.array(fails)
    isfailure = np.any(~np.isnan(fails), axis=1)
    fails = fails[isfailure]

    return times, isfailure, xpts


def write_vtk_output(surface, times, isfailure, xpts, pts, vecs):
    import vtkhelpers
    from occhelpers.components import OCCBSplineSurface

    OCCBSplineSurface.from_raw_object(surface).to_interpol_bspline_surface().to_vtk().write('surf')
    vtklines = vtkhelpers.multiple_lines_from_grids(np.dstack((pts,
                                                               pts + vecs * 8)))

    data = {'id': np.arange(len(pts)),
            'failure': isfailure.astype(int).ravel(),
            'runtime': np.array(times),
            }

    xverts = vtkhelpers.vertices_from_grids(xpts.T)
    xverts.point_data.update(data)
    xverts.write('xpoints')

    vtklines.cell_data.update(data)
    vtklines.write('lines')


if __name__ == "__main__":
    import pytest
    pytest.main(['test_intersection.py', '-sv'])
