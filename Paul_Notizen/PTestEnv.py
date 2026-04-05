import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class PTestEnv(BaseRLAviary):
    """Single agent RL problem: fly to a target with custom obstacles."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=np.array([[0.0, 0.0, 0.5]]),
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
    ):
        self.TARGET_POS = np.array([0.0, 0.0, 1.2])
        self.EPISODE_LEN_SEC = 8

        self.CUSTOM_OBSTACLES = [
            {
                "pos": np.array([0.5, 0.0, 1.0]),
                "half_extents": np.array([0.1, 0.3, 0.3]),
            }
        ]

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

    def _addObstacles(self):
        """Add custom static obstacles to the PyBullet world."""
        self.OBSTACLE_IDS = []

        for obs in self.CUSTOM_OBSTACLES:
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=obs["half_extents"]
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=obs["half_extents"],
                rgbaColor=[1, 0, 0, 1]
            )
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=obs["pos"]
            )
            self.OBSTACLE_IDS.append(body_id)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        dist_to_target = np.linalg.norm(self.TARGET_POS - pos)
        return -dist_to_target

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        return np.linalg.norm(self.TARGET_POS - pos) < 0.15

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]

        if abs(pos[0]) > 2.5 or abs(pos[1]) > 2.5 or pos[2] > 2.5 or pos[2] < 0.05:
            return True

        if abs(state[7]) > 0.5 or abs(state[8]) > 0.5:
            return True

        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {"answer": 42}