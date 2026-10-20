from gymnasium import Wrapper
from gym_mf_envs.envs.base_env.base_env import BaseMFEnv
from gym_mf_envs.envs.base_env.base_env_infinite import BaseMFInfiniteEnv


class DoneWrapper(Wrapper):
    """
    Abstract base class for done condition wrappers in finite population mean-field environments.
    Subclasses should implement `_check_done(info)` to determine when an episode is considered done.
    """
    def __init__(self, env: BaseMFEnv):
        """
        Initialize the DoneWrapper.
        """
        super(DoneWrapper, self).__init__(env)

    def step(self, action):
        """
        Step through the environment, updating the 'done' condition using the custom logic.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        done = self._check_done(info)
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        """
        return self.env.reset(seed=seed, options=options)

    def _check_done(self, info):
        """
        Custom done condition.
        """
        raise NotImplementedError


class AnyPopulationAllAgentDoneWrapper(DoneWrapper):
    """
    Done if all agents in any one population are done.
    """
    def __init__(self, env: BaseMFEnv):
        """Initialize this object and its environment state."""
        super(AnyPopulationAllAgentDoneWrapper, self).__init__(env)

    def _check_done(self, info):
        """Handle the check done operation for this environment."""
        done_dict = info["done_dict"]
        team_done_flag = [all(done_dict[pop_id].values()) for pop_id in done_dict.keys()]
        return any(team_done_flag)


class AllDoneWrapper(DoneWrapper):
    """
    Done only when all agents in all populations are done.
    """
    def __init__(self, env: BaseMFEnv):
        """Initialize this object and its environment state."""
        super(AllDoneWrapper, self).__init__(env)

    def _check_done(self, info):
        """Handle the check done operation for this environment."""
        done_dict = info["done_dict"]
        team_done_flag = [all(done_dict[pop_id].values()) for pop_id in done_dict.keys()]
        return all(team_done_flag)


class DoneInfiniteWrapper(Wrapper):
    """
    Abstract base class for done condition wrappers in infinite population mean-field environments.
    """
    def __init__(self, env: BaseMFInfiniteEnv):
        """
        Initialize the DoneInfiniteWrapper.
        """
        super(DoneInfiniteWrapper, self).__init__(env)

    def step(self, action):
        """
        Step through the environment and compute done condition.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        done = self._check_done(info)
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        """
        return self.env.reset(seed=seed, options=options)

    def _check_done(self, info):
        """
        Abstract method to compute the custom done condition.
        """
        raise NotImplementedError

class AllDoneInfiniteWrapper(DoneInfiniteWrapper):
    """
    Done when the 'done_dict' indicates all populations are done.
    """
    def __init__(self, env:  BaseMFInfiniteEnv):
        """Initialize this object and its environment state."""
        super(AllDoneInfiniteWrapper, self).__init__(env)

    def _check_done(self, info):
        """Handle the check done operation for this environment."""
        done_dict = info["done_dict"]
        team_done_flag = [all(done_dict.values())]
        return all(team_done_flag)
