import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
import torch as th
from gymnasium import spaces
import matplotlib.pyplot as plt


from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=380000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=100000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--epsilon", type=str, default=0.05,
        help="the epsilon-greedy parameter")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()

    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward

class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def _maybe_cast_dtype(self, dtype: Any) -> np.dtype:
        """
        Helper method to cast the action space's data type to a compatible format.

        :param dtype: The data type of the action space.
        :return: A compatible numpy data type.
        """
        if isinstance(dtype, np.dtype):
            return dtype
        return np.dtype(dtype)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape to handle multi-dim and discrete action spaces
        if len(self.actions.shape) > 2:
            action = action.reshape(self.actions.shape[2:])

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        # Update position and handle buffer full condition
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        data = (
            self.observations[batch_inds, 0],
            self.actions[batch_inds, 0],
            self.next_observations[batch_inds, 0],
            self.dones[batch_inds, 0],
            self.rewards[batch_inds, 0],
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)

# Evaluation
def evaluate_model(q_network, env, num_episodes=100):
    q_network.eval()  # Set to evaluation mode
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                action = q_network(torch.Tensor(obs).to(device)).argmax(dim=1).cpu().numpy()
            obs, reward, terminations, truncations, _ = env.step(action)
            episode_reward += reward.item()
            done = terminations.any() or truncations.any()
        
        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Evaluation over {num_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device)
    start_time = time.time()

    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_update_frequency = 1000  # 每1000步更新一次目标网络

    # Training loop
    obs, _ = envs.reset(seed=args.seed)
    episode_rewards = []  # Store episodic rewards for plotting
    steps = []  # Store steps for plotting
    for global_step in range(args.total_timesteps):
        print(f"global_step: {global_step}/{args.total_timesteps}")
        # Epsilon-greedy exploration
        epsilon = args.epsilon
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = q_network(torch.Tensor(obs).to(device)).argmax(dim=1).cpu().numpy()

        # Execute the game
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_rewards.append(info["episode"]["r"])
                    steps.append(global_step)
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

        # Save data to replay buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Training
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                # target_max = q_network(data.next_observations).max(dim=1)[0]
                target_max = target_network(data.next_observations).max(dim=1)[0]
                td_target = data.rewards + args.gamma * (1 - data.dones) * target_max
            old_val = q_network(data.observations).gather(1, data.actions.long()).squeeze()
            loss = nn.functional.mse_loss(td_target, old_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新目标网络
            if global_step % target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    # Save model
    model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    torch.save(q_network.state_dict(), model_path)
    print(f"model saved to {model_path}")

    # Plot and save the reward vs. step graph

    # Create smoothed rewards using moving average
        # Plot and save the reward vs. step graph
    def moving_average(data_list, window_size=20):
        data = np.array(data_list, dtype=np.float32)
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window = data[start_idx:(i + 1)]
            smoothed.append(np.mean(window))
        return np.array(smoothed)

    if episode_rewards and steps:
        # Convert lists to numpy arrays
        rewards_array = np.array(episode_rewards)
        steps_array = np.array(steps)
        
        # Calculate smoothed rewards
        window_size = min(20, len(rewards_array))  # Smaller window size
        smoothed_rewards = moving_average(rewards_array, window_size)

        plt.figure(figsize=(10, 6))
        # Plot raw rewards in light red
        plt.plot(steps_array, rewards_array, 'lightcoral', alpha=0.3, label='Raw Reward')
        # Plot smoothed rewards in red
        plt.plot(steps_array, smoothed_rewards, 'red', linewidth=2, label='Smoothed Reward')
        
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward vs. Step")
        plt.legend()
        plt.grid(True)

        # Save the plot to the same folder as the model
        plot_path = f"runs/{run_name}/reward_vs_step.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")
    else:
        print("No reward data to plot")

    # Save hyperparameters to a txt file
    hyperparams_path = f"runs/{run_name}/hyperparameters.txt"
    with open(hyperparams_path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"Hyperparameters saved to {hyperparams_path}")

    # model_path = r"D:\College\Grade4_Fall\DDA4230\Assignment\runs\CartPole-v1__dqn_tostu_250316ds__1__1742120084\dqn_tostu_250316ds.cleanrl_model"
    # model_path = r"D:\College\Grade4_Fall\DDA4230\Assignment\runs\CartPole-v1__dqn_tostu_250316ds__1__1742121675\dqn_tostu_250316ds.cleanrl_model"
    # Evaluation
    q_network.load_state_dict(torch.load(model_path))
    mean_reward, std_reward = evaluate_model(q_network, envs)
    print(f"Evaluation reward: {mean_reward:.2f} +/- {std_reward:.2f}")


    # q_network.load_state_dict(torch.load(model_path))
    # obs, _ = envs.reset()
    # total_rewards = 0
    # for _ in range(10000):
    #     with torch.no_grad():
    #         actions = q_network(torch.Tensor(obs).to(device)).argmax(dim=1).cpu().numpy()
    #     obs, rewards, terminations, truncations, infos = envs.step(actions)
    #     total_rewards += rewards.sum()
    #     if terminations.any():  # 如果 episode 终止，重置环境
    #         obs, _ = envs.reset()
    # print(f"Evaluation reward: {total_rewards / 10000}")
