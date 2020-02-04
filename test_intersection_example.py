import pytest

from intersection_example import *

BREP_INPUTS = ['ellipsoid.brep']


@pytest.fixture(params=BREP_INPUTS)
def surface(request):
    return read_brep(request.param)


@pytest.fixture
def points_and_vectors():
    return generate_intersection_vectors(40., nxu=50, nxv=50)


@pytest.fixture
def lines(points_and_vectors):
    return trimmed_curve_from_points(*points_and_vectors)


@pytest.fixture
def intersections(surface, lines, points_and_vectors):
    return intersect(surface, lines, points_and_vectors)


def test_intersections(intersections):
    _, isfailure, _ = intersections
    assert not np.sum(isfailure)


def test_plot_vtk(surface, points_and_vectors, intersections):
    try:
        times, isfailure, xpts = intersections
        pts, vecs = points_and_vectors

        write_vtk_output(surface, times, isfailure, xpts, pts, vecs)
    except ImportError:
        pytest.skip()


def test_plot_matplotlib(intersections, points_and_vectors):
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    times, isfailure, xpts = intersections
    v = np.column_stack(points_and_vectors)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    p = ax.scatter(*xpts.T, c=times, cmap='Blues')
    bar = plt.colorbar(p)
    bar.set_label('runtime $t/s$')

    ax.plot(*xpts[isfailure].T, marker='o', color='r', ls='', alpha=0.8)
    ax.quiver(*v[isfailure].T, color='r', alpha=0.8, pivot='tail', length=6.)

    plt.show()


if __name__ == "__main__":
    pytest.main([__file__ + '::test_intersections'])
