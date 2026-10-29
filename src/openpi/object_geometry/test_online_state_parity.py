"""Training-side disk state and deployment-side array state must be equivalent."""
import numpy as np

from openpi.object_geometry.production import CausalHandSceneState


class _Provider:
    def __init__(self):
        self.force = None

    def get_force_norm(self, episode, frame):
        return self.force


def test_online_offline_state_same_current_arrays_and_reset():
    provider = _Provider()
    disk = CausalHandSceneState(spatial_tactile=provider)
    online = CausalHandSceneState()
    first = np.linspace(0, 0.6, 600, dtype=np.float32)
    for frame, force in ((0, first), (1, first + .2), (3, None), (4, first + .1)):
        provider.force = force
        a = disk.next(episode=0, frame=frame)
        b = online.next_from_arrays(frame=frame, force_norm=force)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    online.reset()
    assert online._previous_frame is None
    reset_first = online.next_from_arrays(frame=0, force_norm=first)
    np.testing.assert_array_equal(reset_first["grasp_short_delta_force_norm"], np.zeros(600))
