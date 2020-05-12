import numpy as np
import pytest

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.Geom import Geom_Line, Geom_TrimmedCurve
from OCC.Core.GeomConvert import geomconvert_SurfaceToBSplineSurface
from OCC.Core.GeomAPI import GeomAPI_ExtremaCurveSurface
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.TopoDS import TopoDS_Shape, topods_Face


ELLIPSOID_INPUT = 'ellipsoid.brep'
CURVE_POINTS_INPUT = 'curve_points.csv'
CURVE_VECTORS_INPUT = 'curve_vectors.csv'
EXPECTED_POINTS_INPUT = 'expected_points.csv'


@pytest.fixture
def ellipsoid():
    """
    Read ellipsoid surface from brep file
    """
    output_shape = TopoDS_Shape()
    builder = BRep_Builder()
    breptools_Read(output_shape, ELLIPSOID_INPUT, builder)
    brep_face = BRep_Tool.Surface(topods_Face(output_shape))
    return geomconvert_SurfaceToBSplineSurface(brep_face)


@pytest.fixture
def curves():
    """
    Generate the intersection vector curves
    """
    points = np.loadtxt(CURVE_POINTS_INPUT, delimiter=',')
    vectors = np.loadtxt(CURVE_VECTORS_INPUT, delimiter=',')

    return [Geom_TrimmedCurve(Geom_Line(gp_Pnt(*point), gp_Dir(*vector)), 0., 1.)
            for point, vector in zip(points, vectors)]


@pytest.fixture
def expected_points():
    return np.loadtxt(EXPECTED_POINTS_INPUT, delimiter=',')


def test_ellipsoid_intersection(ellipsoid, curves, expected_points):
    """
    Perform intersection and compare output to expected results
    """
    xpts = []
    fails = []

    for curve in curves:
        x = GeomAPI_ExtremaCurveSurface(curve, ellipsoid, 0., np.inf, 0., 1., 0., 1.)
        w, u, v = x.LowerDistanceParameters()
        vec = curve.Value(w)
        if x.LowerDistance() > 1e-4:
            fails.append(True)
        xpts.append([vec.X(), vec.Y(), vec.Z()])
        fails.append(False)

    xpts = np.array(xpts)
    isfailure = np.array(fails)

    assert not np.sum(isfailure)
    assert np.allclose(xpts, expected_points)


if __name__ == "__main__":
    pytest.main([__file__, '-sv'])
