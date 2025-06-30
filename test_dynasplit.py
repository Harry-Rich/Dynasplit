import numpy as np
import MDAnalysis as mda
import Dynasplit as ds  # Replace with your actual module name if different
import pytest


@pytest.fixture
def u():
    return mda.Universe('./test_traj/290k_end.data', './test_traj/sampled_100_frames.dcd')  # Tiny test files

def test_dynasplit_rotational_and_translational_consistency(u):
    # Load the Universe

    dimensions = u.dimensions[:3]

    # Standard decomposition
    splitter = ds.Dynasplit(u)
    splitter.decompose()

    # Slower decomposition with specified atom types
    slower = ds.Dynasplit(u)
    slower.decompose(slower=True, atom_types='type 1 2 3 4 5 6 7 8 9 10 11 12')

    # Rotational trajectory consistency check
    assert np.isclose(
        slower.rot_traj % dimensions,
        splitter.rot_traj % dimensions,
        atol=5e-5
    ).all(), "Rotational trajectories differ beyond tolerance."

    # Translational trajectory consistency check
    assert np.isclose(
        slower.trans_traj % dimensions,
        splitter.trans_traj % dimensions,
        atol=5e-5
    ).all(), "Translational trajectories differ beyond tolerance."


def test_pseudo_com_method_basic():
    coords = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])  
    indices = np.array([[0, 1]])
    masses = np.array([[1, 1]])
    box = np.array([1.0, 1.0, 1.0])
    result = ds.pseudo_com_method(coords, indices, masses, box)
    assert result.shape == (1, 3)
    assert np.all(result >= 0) and np.all(result <= box)
    assert (result == [[0,0,0.5]]).all()


