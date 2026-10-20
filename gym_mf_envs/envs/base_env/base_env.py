from typing import Any
import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces
import imageio
from pathlib import Path
import copy
from gym_mf_envs.envs.base_env.base_oracle import BaseMFOracle
from gym_mf_envs.utils import flat_dict_values


class BaseMFEnv(gym.Env):
    """The base class of the Mean-Field Multi-Agent Environment. It encapsulates
    an environment with arbitrary behind-the-scenes dynamics. This environment
    can be partially observed for all populations.

    The main API methods that users of this class need to know are:

        - step
        - reset
        - render
        - close
        - seed
        - load_config       <- new method   #todo to be added

    And set the following attributes:
        action_space[population_id]:
            A Space object corresponding to valid actions of all homogenous agents within population_id
        observation_space[population_id]["local-obs"][observation_name]:
            A Space object corresponding to private obs with observation_name within population_id
        observation_space[population_id]["shared-team-obs"][observation_name]:
            A Space object corresponding to team-shared obs with observation_name within population_id
        observation_space["global-obs"][observation_name]:
            A Space object corresponding to the global observation shared among all agents in the game


    The methods are accessed publicly as "step", "reset", etc...
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }

    def __init__(self, render_mode=None, num_population: int = 1, num_agent_list: dict[str, int] = {"blue": 1},
                 oracle_kwargs: dict[str, Any] = {"rho": 0.5}):
        """Initialize this object and its environment state."""
        super().__init__()
        self.window_size = 512

        # setup agents and populations
        self._num_population = num_population
        self._num_agents_dict = num_agent_list
        assert len(self._num_agents_dict) == self._num_population

        # set up observation space and init observation dict
        self.observation_space = spaces.Dict(
            {
                population_id:
                    spaces.Dict(
                        {"agent_{}-local-obs".format(agent_index): spaces.Space()
                         for agent_index in range(num_agents)
                         })
                for population_id, num_agents in self._num_agents_dict.items()
            } |
            {"global-obs": spaces.Space()}
        )
        self._setup_observation_space()

        # set up action space
        self.action_space = spaces.Dict(
            {
                population_id:
                    spaces.Space()
                for population_id in self._num_agents_dict.keys()
            }
        )

        self._setup_action_space()

        # initialize agents and mean field information. These information attributes contain the system state
        self.agent_info = {
            population_id:
                {
                    "agent_{}".format(agent_index):
                        self._init_agent_info(population_id, agent_index)
                    for agent_index in range(num_agents)
                } for population_id, num_agents in self._num_agents_dict.items()
        }
        self.mf_info: dict[str, np.ndarray(dtype=np.float32)] = self._init_mf_info()

        # initialize environment info (e.g. obstacles, targets)
        self.env_info = self._init_env_info()

        # initialize mean-field oracle
        self.mf_oracle = self._setup_mf_oracle(**oracle_kwargs)

        # setup time step tracker
        self._time_step = 0

        # render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.frame_count = 0
        self.frames = []

        self.window = None
        self.clock = None

    @property
    def population_id_list(self):
        """Return the configured population identifiers."""
        return list(self._num_agents_dict.keys())

    def reset(self, seed=None, options=None):
        """Reset episode state and return the initial observation and info."""
        super().reset(seed=seed)
        
        info = self.info_dict if hasattr(self, 'info_dict') else self._get_info()
        info = copy.deepcopy(info)
        
        self.action_space.seed(seed=seed)

        self._reset_specific_env_variables()

        self._reset_mf_oracles()

        self._random_init_agents()

        self.mf_oracle.update_mf_info(agent_info=self.agent_info, mf_info=self.mf_info)

        observations = self._get_obs()
        
        if self.render_mode == "human":
            self._render_frame()

        self._time_step = 0

        return observations, info

    def step(self, actions: dict[dict]):
        # actions[population_id][agent_id] gives the action of an agent
        """Advance the environment by one joint action and return transition results."""
        self._update_agent_info(actions)
        self.mf_oracle.update_mf_info(agent_info=self.agent_info, mf_info=self.mf_info)

        reward_dict = self.mf_oracle.get_reward(self.agent_info, self.mf_info)
        observations_dict = self._get_obs()

        done_dict = self._check_agent_termination()

        truncated = False
        self.info_dict = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        self._time_step += 1

        assert "done_dict" in self.info_dict.keys()
        return observations_dict, reward_dict, done_dict, truncated, self.info_dict

    def demo(self, **kwargs):
        """Run the environment’s built-in demonstration."""
        raise NotImplemented

    def sample_actions(self, action_sample_type):
        """Sample one action for each population or agent."""
        action_dict = {
            population_id:
                {
                    agent_id: self.action_space[population_id][agent_id].sample()
                    for agent_id in self.action_space[population_id].keys()
                }
            for population_id in self.action_space.keys()
        }
        return action_dict

    def _get_obs(self):
        """Build and return the current observation structure."""
        raise NotImplemented

    def _get_info(self):
        """Build and return auxiliary information for the current state."""
        raise NotImplemented

    def _update_agent_info(self, actions):
        """Apply actions and update each agent’s state information."""
        raise NotImplemented

    def _check_agent_termination(self) -> dict[str, dict[str, bool]]:
        """Return termination flags for individual agents."""
        raise NotImplemented

    def _render_frame(self):
        """Render the current environment frame."""
        raise NotImplemented

    def _reset_specific_env_variables(self):
        """Reset environment-specific state for a new episode."""
        raise NotImplemented

    def _random_init_agents(self):
        """Place agents at valid initial locations."""
        raise NotImplemented

    def _reset_mf_oracles(self):
        """Reset the mean-field oracle state."""
        raise NotImplemented

    def _init_agent_info(self, population_id, agent_id):
        """Create initial information for one agent."""
        raise NotImplemented

    def _init_mf_info(self):
        """Create the initial mean-field state."""
        raise NotImplemented

    def _init_env_info(self):
        """Create initial environment-level information."""
        raise NotImplemented

    def _init_observations(self):
        """Handle the init observations operation for this environment."""
        raise NotImplemented

    def _setup_mf_oracle(self, **kwargs) -> BaseMFOracle:
        """Construct the oracle used for mean-field rewards and transitions."""
        raise NotImplemented

    def _setup_observation_space(self):
        """Define the Gymnasium observation space."""
        raise NotImplemented

    def _setup_action_space(self):
        """Define the Gymnasium action space."""
        raise NotImplemented

    def get_population_size(self, population_id):
        """Return the number of agents in a population."""
        return len(self._num_agents_dict[population_id])

    def render(self):
        """Render the current environment state."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        """Release resources held by the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_n_agents(self, population_id):
        """Return the number of agents in a population."""
        return self._num_agents_dict[population_id]

    def get_population_ids(self):
        """Return the configured population identifiers."""
        return self._num_agents_dict.keys()
    
    def display_gif(self, duration=0.08, loop=0, title="MF-Envs demo", caption=""):
        """Encode captured frames as a GIF and display it in a styled IPython card."""
        import base64
        from html import escape
        from io import BytesIO
        from IPython.display import HTML, display

        gif_buffer = BytesIO()
        try:
            imageio.mimsave(
                gif_buffer,
                self.frames,
                format="GIF",
                duration=duration,
                loop=loop,
            )
        finally:
            self.close()

        gif_data = base64.b64encode(gif_buffer.getvalue()).decode("ascii")
        html = """
        <div style="max-width:760px;margin:20px auto;padding:18px 20px 20px;
                    background:#ffffff;border:1px solid #e2e8f0;border-radius:18px;
                    box-shadow:0 12px 32px rgba(15,23,42,.10);font-family:system-ui,-apple-system,sans-serif;">
          <div style="margin:0 0 14px 2px;">
            <div style="color:#0f172a;font-size:19px;font-weight:700;letter-spacing:-.02em;">{title}</div>
            <div style="color:#64748b;font-size:13px;margin-top:4px;">{caption}</div>
          </div>
          <img src="data:image/gif;base64,{gif_data}"
               style="display:block;width:100%;height:auto;border-radius:12px;background:#f8fafc;"
               alt="{title} animated environment visualization" />
        </div>
        """
        html = html.replace("{title}", escape(title))
        html = html.replace("{caption}", escape(caption))
        html = html.replace("{gif_data}", gif_data)
        display(HTML(html))

    def save_gif(self, video_dir):
        """Save recorded rendered frames as an animated GIF."""
        print("Entering here")
        video_dir = Path(video_dir)  
        filename = 'video.gif'
        filepath = video_dir / filename
        print("length of frames", len(self.frames))
        print(filepath)
        imageio.mimsave(filepath, self.frames, duration=50)


