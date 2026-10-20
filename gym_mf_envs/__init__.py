from gymnasium.envs.registration import register

disable_passive_checker = False

register(
    id="gym_mf_envs/MF-Nav-v0",
    entry_point="gym_mf_envs.envs.grid_envs.envs.cooperative_team_envs:MFGridNavEnv",
    max_episode_steps=1000,
    disable_env_checker=disable_passive_checker
)

register(
    id="gym_mf_envs/MF-Battle-v0",
    entry_point="gym_mf_envs.envs.grid_envs.envs.competitive_team_envs:MFBattleFieldEnv",
    max_episode_steps=300,
    disable_env_checker=disable_passive_checker
)

register(
    id="gym_mf_envs/MF-Virus-v0",
    entry_point="gym_mf_envs.envs.grid_envs.envs.competitive_team_envs:MFVirusEnv",
    max_episode_steps=300,
    disable_env_checker=disable_passive_checker
)

register(
    id="gym_mf_envs/MF-RPSMFEnv-v0",
    entry_point="gym_mf_envs.envs.benchmark_envs.envs.rock_paper_scissor:RPSMFEnv",
    max_episode_steps=100,
    disable_env_checker=disable_passive_checker
)

register(
    id="gym_mf_envs/MF-Battle-Infinite-v0",
    entry_point="gym_mf_envs.envs.grid_envs.envs.competitive_team_envs_infinite:MFBattleFieldInfiniteEnv",
    max_episode_steps=300,
    disable_env_checker=disable_passive_checker
)

register(
    id="gym_mf_envs/MF-Virus-Infinite-v0",
    entry_point="gym_mf_envs.envs.grid_envs.envs.competitive_team_envs_infinite:MFVirusInfiniteEnv",
    max_episode_steps=300,
    disable_env_checker=disable_passive_checker
)