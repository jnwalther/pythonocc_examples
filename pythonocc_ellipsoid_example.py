from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import timeit

import mpl_toolkits.mplot3d
from OCC.Geom import Geom_Line, Geom_BSplineSurface, Geom_TrimmedCurve
from OCC.GeomAPI import GeomAPI_ExtremaCurveSurface
from OCC.gp import gp_Pnt, gp_Dir
from OCC.TColgp import TColgp_Array2OfPnt
from OCC.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger, TColStd_Array2OfReal
from OCC.Display.SimpleGui import init_display


def line_from_points(points=[[0.9, 0.9, 0.9]], vecs=[[1, 1, 0]]):
    '''
    Create line objects from points/vectors
    '''
    return [Geom_TrimmedCurve(Geom_Line(gp_Pnt(*p), gp_Dir(*vec)).GetHandle(), 0., 6.).GetHandle()
            for p, vec in zip(points, vecs)]


def get_pythonocc_bspline_surface(surf_input, use_weights=False):
    '''
    Create bspline surface objects from stored data
    '''
    try:
        return surface_from_interpolation(surf_input)
    except (ImportError, TypeError):
        with open('surf_data_py%s.p' % sys.version_info[0], 'rb') as f:
            data = pickle.load(f)

    pts_ = data['pts_']
    deg = data['deg']
    knots = data['knots']
    mults = data['mults']
    periodic = data['periodic']

    udeg, vdeg = deg
    uperiod, vperiod = periodic
    cpts = TColgp_Array2OfPnt(1, pts_.shape[0], 1, pts_.shape[1])
    weights = TColStd_Array2OfReal(1, pts_.shape[0], 1, pts_.shape[1])

    for i, pts in enumerate(pts_):
        for j, pt in enumerate(pts):
            cpts.SetValue(i+1, j+1, gp_Pnt(*pt))
            weights.SetValue(i+1, j+1, float(1))

    uknots = TColStd_Array1OfReal(1, knots[0].shape[0])
    for i, val in enumerate(knots[0]):
        uknots.SetValue(i+1, val)
    vknots = TColStd_Array1OfReal(1, knots[1].shape[0])
    for i, val in enumerate(knots[1]):
        vknots.SetValue(i+1, val)

    umult = TColStd_Array1OfInteger(1, mults[0].shape[0])
    for i, val in enumerate(mults[0]):
        umult.SetValue(i+1, int(val))
    vmult = TColStd_Array1OfInteger(1, mults[1].shape[0])
    for i, val in enumerate(mults[1]):
        vmult.SetValue(i+1, int(val))

    if use_weights:
        return Geom_BSplineSurface(cpts, weights, uknots, vknots, umult, vmult,
                                   int(udeg), int(vdeg), uperiod, vperiod).GetHandle()
    else:
        return Geom_BSplineSurface(cpts, uknots, vknots, umult, vmult,
                                   int(udeg), int(vdeg), uperiod, vperiod).GetHandle()


def show_objects(objs=[], wireframe=False):
    '''
    Display pythonOCC geometry objects in simple viewer

    Parameters
    ----------
    objs : list
        List of pythonOCC objects to be displayed
    wireframe : bool
        Set to true to plot wireframe instead of surface
    '''
    display, start_display, add_menu, add_function_to_menu = init_display(size=(1920, 1080))
    display.EraseAll()
    if wireframe:
        display.SetModeWireFrame()
    for obj in objs:
        display.DisplayShape(obj, update=True)
    start_display()


def surface_from_interpolation(input_array):
    '''
    Generate
    '''
    from bt_tools.bt_math import interpol_surface
    from pythonocc_geometry_tools import surfs_from_bsplines
    surf = interpol_surface.BSplineSurface.from_interpolation_points(input_array)
    return surfs_from_bsplines([surf], dump=1)[0]


def generate_ellipsoid(a, b, c, nu=40, nv=60, ax=None):
    '''
    Generate ellipsoid profiles

    Output serves as input for interpolation routine (not provided)
    '''
    u = np.linspace(-np.pi/2, np.pi/2, nu)
    v = np.linspace(0, 2*np.pi, nv)
    profs = []
    for u_ in u:
        x = c*np.sin(u_)*np.ones(v.shape)+c
        y = a*np.cos(u_)*np.cos(v)
        z = b*np.cos(u_)*np.sin(v)
        profs.append([[x, y, z]])
#        if ax is not None:
#            ax.plot(x, y, z)

    return np.array(profs).squeeze().swapaxes(1, 2)


def generate_intersection_vectors(c, nxu=100, nxv=10):
    '''
    Generation routine for the intersection vectors
    '''
    vec_x = np.linspace(0.005*c, 1.995*c, nxu)
    vec_phi = np.linspace(0, 2*np.pi, nxv)
    v0 = np.pad(vec_x.reshape(-1, 1), ((0, 0), (0, 2)), mode='constant')

    return np.vstack([np.hstack([v0,
                                 np.vstack([np.zeros(vec_x.shape),
                                            np.zeros(vec_x.shape)+np.cos(phi),
                                            np.zeros(vec_x.shape)+np.sin(phi)]).T])
                      for phi in vec_phi])


def intersect(surf_input, v, ax=None, show_occ=False):
    '''
    Actual intersection
    '''
    occsurf = get_pythonocc_bspline_surface(surf_input)
    occvecs = line_from_points(v[:, :3], v[:, 3:])

    occpts = []
    xpts = []
    fails = []
    times = []
    for curve, v_1, v_2 in zip(occvecs, v[:, :3], v[:, 3:]):
        failure = [np.nan, np.nan, np.nan]
        t = timeit.timeit(lambda: GeomAPI_ExtremaCurveSurface(curve, occsurf, 0., 1., 0., 1., 0., 1.),
                          number=1)
        x = GeomAPI_ExtremaCurveSurface(curve, occsurf, 0., 6., 0., 1., 0., 1.)
        w, _, _ = x.LowerDistanceParameters()
        times.append(t)
        vec = curve.GetObject().Value(w)
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
    print(len(fails), 'failures')

    if ax is not None:
        p = ax.scatter(*xpts.T, c=times, cmap='Blues')
        ax.plot(*fails.T, marker='o', color='k', ls='', alpha=0.8)
        bar = plt.colorbar(p)
        bar.set_label('runtime $t/s$')

    return times, isfailure, xpts

#fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = None
basis_profiles = generate_ellipsoid(5., 5., 40., ax=ax)


def intersect_all_vectors(ax=None, print_analysis=False):
    v = generate_intersection_vectors(40., nxu=50, nxv=50)
    t, fail, xpts = intersect(basis_profiles, v, ax=ax)

    if print_analysis:
        import pandas as pd
        df = pd.DataFrame(zip(t, list(fail.flat)), columns=['time', 'isfailure'])
        g = df.groupby('isfailure')
        print(g.mean())
        print(g.min())
        print(g.max())
        print('===========')
        print(g.sum().sum())

        if ax is not None:
            ax.plot(*xpts[df.time.nlargest(15).index].T, marker='o', color='r', ls='', alpha=0.8)
            ax.quiver(*v[df.time.nlargest(15).index].T, color='r', alpha=0.8, pivot='tail', length=6.)


def intersect_failing_vector(show_occ=False):
    failing_vector = np.array([[0.2, 0., 0., 0., 1., 0.]])
    intersect(basis_profiles, failing_vector, show_occ=show_occ)


def intersect_passing_vector(show_occ=False):
    passing_vector = np.array([[13.19591837, 0., 0., 0., -0.8380881, -0.5455349]])
    intersect(basis_profiles, passing_vector, show_occ=show_occ)


if __name__ == "__main__":
    intersect_all_vectors(ax=ax, print_analysis=1)
    intersect_failing_vector()
    intersect_passing_vector()
#    plt.show()
