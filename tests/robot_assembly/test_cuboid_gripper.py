"""Part-width aperture shaping must not alter recorded-action replay."""

import numpy as np
import warp as wp

import castle_reasoning.runtime.control.gripper as gripper


def test_aperture_mapping_and_replay_bypass():
    wp.init()
    action = np.zeros((3, 7), np.float32)
    action[:, 6] = [-1, 0, 1]
    device_action = wp.array(action, dtype=float, device="cpu")
    enabled = wp.array([1], dtype=int, device="cpu")
    index = wp.zeros(3, dtype=int, device="cpu")
    closure = wp.array(np.full((3, 1), 0.6, np.float32), dtype=float, device="cpu")
    wp.launch(
        gripper.shape_aperture,
        3,
        inputs=[enabled, index, closure, device_action],
        device="cpu",
    )
    np.testing.assert_allclose(device_action.numpy()[:, 6], [0.2, 0.6, 1], atol=1e-6)
    recorded = device_action.numpy().copy()
    enabled.zero_()
    wp.launch(
        gripper.shape_aperture,
        3,
        inputs=[enabled, index, closure, device_action],
        device="cpu",
    )
    np.testing.assert_array_equal(device_action.numpy(), recorded)
