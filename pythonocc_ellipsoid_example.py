from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import time

from mpl_toolkits.mplot3d import Axes3D
from OCC.Geom import Geom_Line, Geom_BSplineSurface
from OCC.GeomAPI import GeomAPI_ExtremaCurveSurface
from OCC.gp import gp_Pnt, gp_Dir
from OCC.TColgp import TColgp_Array2OfPnt, TColgp_Array1OfPnt
from OCC.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger

fig = plt.figure()
ax = fig.gca(projection='3d')

### PythonOCC helpers

def line_from_points(points=[[0.9,0.9,0.9]], vecs=[[1,1,0]]):
    '''
    Create line objects from points/vectors
    '''
    return [Geom_Line(gp_Pnt(*p), gp_Dir(*vec)).GetHandle() for p, vec in zip(points, vecs)]

def get_pythonocc_bspline_surface(surf_input):
    '''
    Create bspline surface objects from stored data
    '''
    try:
        return surface_from_interpolation(surf_input)
    except (ImportError, TypeError) as e:
        print('Could not create surface by interpolation. Falling back to stored data in surf_data_py%s.p.'%sys.version_info[0])
        with open('surf_data_py%s.p'%sys.version_info[0], 'rb') as f:
            data = pickle.load(f)

    pts_ = data['pts_']
    deg = data['deg']
    knots = data['knots']
    mults = data['mults']
    periodic = data['periodic']

    udeg, vdeg = deg
    uperiod, vperiod = periodic
    cpts = TColgp_Array2OfPnt(1, pts_.shape[0], 1, pts_.shape[1])

    for i, pts in enumerate(pts_):
        for j, pt in enumerate(pts):
            cpts.SetValue(i+1, j+1, gp_Pnt(*pt))

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

    return Geom_BSplineSurface(cpts, uknots, vknots, umult, vmult, udeg, vdeg, uperiod, vperiod).GetHandle()


### Create geometries

def surface_from_interpolation(input_array):
    '''
    Generate
    '''
    from bt_tools.bt_math import interpol_surface
    from pythonocc_geometry_tools import surfs_from_bsplines
    surf = interpol_surface.BSplineSurface.from_interpolation_points(input_array)
    return surfs_from_bsplines([surf], dump=1)[0]


def generate_ellipsoid(a, b, c, nu=40, nv=60):
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
        profs.append([[x,y,z]])
        ax.plot(x, y, z)

    return np.array(profs).squeeze().swapaxes(1,2)


def generate_intersection_vectors(c, nxu=100, nxv=10):
    '''
    Generation routine for the intersection vectors
    '''
    vec_x = np.linspace(0.005*c ,1.995*c, nxu)
    vec_phi = np.linspace(0, 2*np.pi, nxv)
    v0 = np.pad(vec_x.reshape(-1,1), ((0, 0),(0, 2)), mode='constant')

    return np.vstack([np.hstack([v0, np.vstack([np.zeros(vec_x.shape), np.zeros(vec_x.shape)+np.cos(phi), np.zeros(vec_x.shape)+np.sin(phi)]).T]) for phi in vec_phi])

###  Intersect

def intersect(surf_input, v):
    '''
    Actual intersection
    '''
    occsurf = get_pythonocc_bspline_surface(surf_input)
    occvecs = line_from_points(v[:,:3], v[:,3:])

    xpts = []
    fails = []
    t = time.time()
    for curve in occvecs:
        failure = [np.nan, np.nan, np.nan]
        x = GeomAPI_ExtremaCurveSurface(curve, occsurf, 0., 6., 0., 1., 0., 1.)
        w, u, v = x.LowerDistanceParameters()
        vec = curve.GetObject().Value(w)
        if x.LowerDistance()>1e-4:
            failure = np.array([vec.X(), vec.Y(), vec.Z()])
        xpts.append(np.array([vec.X(), vec.Y(), vec.Z()]))
        fails.append(failure)

    print('Time for intersections', time.time()-t)

    xpts = np.array(xpts)
    fails = np.array(fails)
    fails = fails[np.any(~np.isnan(fails), axis=1)]
    print(fails)
    print(len(fails), 'failures')

    ax.plot(*xpts.T, marker='o', ls='')
    ax.plot(*fails.T, marker='o', color='k', ls='')

    plt.show()


if __name__ ==  "__main__":
    basis_profiles = generate_ellipsoid(5.,5.,40.) #just for show, using stored BSpline surface data from pickle
    v = generate_intersection_vectors(40., nxu=50, nxv=50)
    intersect(basis_profiles, v)
