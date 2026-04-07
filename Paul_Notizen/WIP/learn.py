"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

This version is adapted to train and test the custom PTestEnv environment.
It can continue training from latest_model.zip if that file already exists.
"""

import os
import time
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.PTestEnv import PTestEnv
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType("kin")
DEFAULT_ACT = ActionType("vel")
DEFAULT_AGENTS = 2
DEFAULT_MA = False

LATEST_MODEL_PATH = "latest_model.zip"


def run(
    multiagent=DEFAULT_MA,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    gui=DEFAULT_GUI,
    plot=True,
    colab=DEFAULT_COLAB,
    record_video=DEFAULT_RECORD_VIDEO,
    local=True,
):
    filename = os.path.join(output_folder, "save-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename + "/")

    if not multiagent:
        train_env = make_vec_env(
            PTestEnv,
            env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=False),
            n_envs=1,
            seed=0,
        )
        eval_env = PTestEnv(obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=False)
    else:
        train_env = make_vec_env(
            MultiHoverAviary,
            env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
            n_envs=1,
            seed=0,
        )
        eval_env = MultiHoverAviary(
            num_drones=DEFAULT_AGENTS,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            gui=False,
        )

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.0 if not multiagent else 949.5
    else:
        target_reward = 467.0 if not multiagent else 920.0

    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename + "/",
        log_path=filename + "/",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    if os.path.isfile(LATEST_MODEL_PATH):
        model = PPO.load(LATEST_MODEL_PATH)
        print("[INFO] Loaded existing model from", LATEST_MODEL_PATH)
    else:
        model = PPO("MlpPolicy", train_env, verbose=1)
        print("[INFO] Created new model")

    model.set_env(train_env)

    total_steps = 15000 if local else 100
    model.learn(
        total_timesteps=total_steps,
        callback=eval_callback,
        log_interval=100,
    )

    model.save(LATEST_MODEL_PATH)
    model.save(filename + "/final_model.zip")
    print("[INFO] Saved latest model to", LATEST_MODEL_PATH)
    print("[INFO] Run folder:", filename)

    eval_file = filename + "/evaluations.npz"
    if os.path.isfile(eval_file):
        with np.load(eval_file) as data:
            timesteps = data["timesteps"]
            results = data["results"][:, 0]
            print("Data from evaluations.npz")
            for j in range(timesteps.shape[0]):
                print(f"{timesteps[j]},{results[j]}")
            if local and len(timesteps) > 0:
                plt.plot(timesteps, results, marker="o", linestyle="-", markersize=4)
                plt.xlabel("Training Steps")
                plt.ylabel("Episode Reward")
                plt.grid(True, alpha=0.6)
                plt.show()

    if os.path.isfile(filename + "/best_model.zip"):
        path = filename + "/best_model.zip"
        print("[INFO] Testing best model from current run")
    else:
        path = LATEST_MODEL_PATH
        print("[INFO] No best_model.zip found, testing latest_model.zip instead")

    model = PPO.load(path)

    if not multiagent:
        test_env = PTestEnv(
            gui=gui,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            record=record_video,
        )
        test_env_nogui = PTestEnv(
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            gui=False,
        )
    else:
        test_env = MultiHoverAviary(
            gui=gui,
            num_drones=DEFAULT_AGENTS,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            record=record_video,
        )
        test_env_nogui = MultiHoverAviary(
            num_drones=DEFAULT_AGENTS,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            gui=False,
        )

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=DEFAULT_AGENTS if multiagent else 1,
        output_folder=output_folder,
        colab=colab,
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        test_env_nogui,
        n_eval_episodes=10,
    )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()

    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        obs2 = obs.squeeze()
        act2 = action.squeeze()

        print(
            "Obs:", obs,
            "\tAction", action,
            "\tReward:", reward,
            "\tTerminated:", terminated,
            "\tTruncated:", truncated,
        )

        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(
                    drone=0,
                    timestamp=i / test_env.CTRL_FREQ,
                    state=np.hstack([
                        obs2[0:3],
                        np.zeros(4),
                        obs2[3:15],
                        act2,
                    ]),
                    control=np.zeros(12),
                )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(
                        drone=d,
                        timestamp=i / test_env.CTRL_FREQ,
                        state=np.hstack([
                            obs2[d][0:3],
                            np.zeros(4),
                            obs2[d][3:15],
                            act2[d],
                        ]),
                        control=np.zeros(12),
                    )

        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)

        if terminated or truncated:
            obs, info = test_env.reset(seed=42, options={})

    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single agent reinforcement learning example script"
    )
    parser.add_argument(
        "--multiagent",
        default=DEFAULT_MA,
        type=str2bool,
        help="Whether to use multi-agent mode (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DEFAULT_RECORD_VIDEO,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--colab",
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    ARGS = parser.parse_args()

    run(**vars(ARGS))