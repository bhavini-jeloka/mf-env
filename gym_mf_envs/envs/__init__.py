"""Environment package with lazily loaded public environment classes."""

from importlib import import_module

_ENVIRONMENT_EXPORTS = {
    "MFGridNavEnv": "gym_mf_envs.envs.grid_envs.envs.cooperative_team_envs",
    "MFBattleFieldEnv": "gym_mf_envs.envs.grid_envs.envs.competitive_team_envs",
    "BaseMFGridEnv": "gym_mf_envs.envs.grid_envs.envs.base_grid_env",
    "RPSMFEnv": "gym_mf_envs.envs.benchmark_envs.envs.rock_paper_scissor",
    "MFBattleFieldInfiniteEnv": "gym_mf_envs.envs.grid_envs.envs.competitive_team_envs_infinite",
}


def __getattr__(name):
    """Load a public environment class only when it is requested."""
    module_name = _ENVIRONMENT_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    return getattr(import_module(module_name), name)


def __dir__():
    """List the environment classes exported by this package."""
    return sorted(set(globals()) | set(_ENVIRONMENT_EXPORTS))


__all__ = list(_ENVIRONMENT_EXPORTS)
