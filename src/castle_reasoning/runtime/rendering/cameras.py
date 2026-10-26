"""Runtime rendering cameras."""

import mujoco
import numpy as np


class Cameras:
    """Two fixed opaque-robot RGB views; no target overlays or state labels."""

    def __init__(self, model, width=512, height=384):
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cameras = []
        for azimuth in (135.0, 45.0):
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [0.425, 0.0, 0.1]
            camera.distance = 0.95
            camera.azimuth = azimuth
            camera.elevation = -35.0
            self.cameras.append(camera)

    def frames(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        frames = []
        for camera in self.cameras:
            self.renderer.update_scene(self.data, camera=camera)
            frames.append(self.renderer.render().copy())
        return np.stack(frames)

    def close(self):
        self.renderer.close()
