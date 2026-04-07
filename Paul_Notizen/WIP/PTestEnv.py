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
        self.TARGET_POS = np.array([1,0,1])
        self.EPISODE_LEN_SEC = 15

        self.CUSTOM_OBSTACLES = []
        #     {
        #         "type": "box",      
        #         "pos": [0, 0, 1],                       #Position Mittelpunkt
        #         "half_extents": [0.1, 0.1, 0.1]         #Halbe groeße
        #     },
        #     {
        #         "type": "sphere",
        #         "pos": [1, 1, 0.5],
        #         "radius": 0.3
        #     },
        #     {"type": "box", "pos": [1, 0.5, 1], "half_extents": [0.05, 0.5, 0.05]},
        #     {"type": "box", "pos": [1, -0.5, 1], "half_extents": [0.05, 0.5, 0.05]},
        #     {"type": "box", "pos": [1, 0, 1.5], "half_extents": [0.5, 0.05, 0.05]},
        #     {"type": "box", "pos": [1, 0, 0.5], "half_extents": [0.5, 0.05, 0.05]}
        # ]

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

    def _addObstacles(self):                                                #Vorgegebene Funktion verstehe noch nciht 100% was da abgeht
        """Add custom static obstacles to the PyBullet world."""
        self.OBSTACLE_IDS = []

        for obs in self.CUSTOM_OBSTACLES:
            if obs["type"] == "box":
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=obs["half_extents"]
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=obs["half_extents"],
                    rgbaColor=[1, 0, 0, 1]
                )

            elif obs["type"] == "sphere":
                collision_shape = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=obs["radius"]
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=obs["radius"],
                    rgbaColor=[1, 1, 0, 1]
                )

            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=obs["pos"]
            )

            self.OBSTACLE_IDS.append(body_id)
        self._addTargetVisual()

    def _addTargetVisual(self):
        self.TARGET_ID = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # kein physisches Objekt
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.1,
                rgbaColor=[0, 1, 0, 1]  # grün
            ),
            basePosition=self.TARGET_POS
        )

    def _hasCollision(self):                                    # Kollisions check
        drone_id = self.DRONE_IDS[0]

        if not hasattr(self, "OBSTACLE_IDS"):
            return False

        for obstacle_id in self.OBSTACLE_IDS:
            contact_points = p.getContactPoints(bodyA=drone_id, bodyB=obstacle_id)
            if len(contact_points) > 0:
                return True

        return False

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        dist_to_target = np.linalg.norm(self.TARGET_POS - pos)

        reward = -dist_to_target

        # kleine Belohnung für Bewegung Richtung Ziel
        if dist_to_target < 1:
            reward += 0.5
        return reward

    def _computeTerminated(self):                       # Ziel erreicht
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        return np.linalg.norm(self.TARGET_POS - pos) < 0.15         # Ist die Drohne näher als 15 cm am Ziel?

    # def _computeObs(self):                              #Kaputt 
    #     state = self._getDroneStateVector(0)
    #     pos = state[0:3]

    #     rel_target = self.TARGET_POS - pos

    #     obs = np.concatenate([state, rel_target])
    #     print(obs)
    #     return obs

    def _computeTruncated(self):                        # Abbruch Fail
        state = self._getDroneStateVector(0)
        pos = state[0:3]

        if abs(pos[0]) > 2.5 or abs(pos[1]) > 2.5 or pos[2] > 2.5 or pos[2] < 0.05:     #abs ist betrag
            return True

        if abs(state[7]) > 0.5 or abs(state[8]) > 0.5:
            return True

        if self._hasCollision():
            return True

        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {"answer": 42}