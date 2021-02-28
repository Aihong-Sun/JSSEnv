from gym.envs.registration import register

register(
    id='jss-v1',
    entry_point='JSSEnv.envs:JssEnv',
)

register(
    id='flexible-jss-v1',
    entry_point='JSSEnv.envs:FlexibleJssEnv',
)