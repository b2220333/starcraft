import sys

from absl import flags
import numpy as np
from pysc2.env import environment
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from deepq.dqn import DQN, load_checkpoint
from common import common

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_SELECTED = features.SCREEN_FEATURES.selected.index
UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'


step_mul = 1
steps = 2000

FLAGS = flags.FLAGS


def main():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(
            map_name="DefeatZerglingsAndBanelings",
            step_mul=step_mul,
            visualize=True,
            game_steps_per_episode=steps * step_mul) as env:

        checkpoint_path = 'models/deepq/checkpoint.pth.tar'
        dqn = DQN()
        dqn, saved_mean_reward = load_checkpoint(dqn, filename=checkpoint_path)
        while True:
            episode_rewards = [0.0]
            obs = env.reset()

            done = False
            player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

            screen = player_relative
            obs, xy_per_marine = common.init(env, obs)

            group_id = 0
            reset = True
            obs, screen, player = common.select_marine(env, obs)
            # step_result = env.step(actions=[
            #     sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
            # ])

            while not done:

                obs, screen, player = common.select_marine(env, obs)
                action = dqn.choose_action(np.array(screen)[None])[0]
                reset = False
                rew = 0

                new_action = None

                obs, new_action = common.marine_action(env, obs, player, action)
                army_count = env._obs[0].observation.player_common.army_count
                try:
                    if army_count > 0 and _ATTACK_SCREEN in obs[0].observation["available_actions"]:
                        obs = env.step(actions=new_action)
                    else:
                        new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
                        obs = env.step(actions=new_action)
                except Exception as e:
                    # print(e)
                    1  # Do nothing

                player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
                new_screen = player_relative

                rew += obs[0].reward

                done = obs[0].step_type == environment.StepType.LAST

                selected = obs[0].observation["screen"][_SELECTED]
                player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()

                if len(player_y) > 0:
                    player = [int(player_x.mean()), int(player_y.mean())]

                if len(player) == 2:

                    if player[0] > 32:
                        new_screen = common.shift(LEFT, player[0] - 32, new_screen)
                    elif player[0] < 32:
                        new_screen = common.shift(RIGHT, 32 - player[0],
                                                  new_screen)

                    if player[1] > 32:
                        new_screen = common.shift(UP, player[1] - 32, new_screen)
                    elif player[1] < 32:
                        new_screen = common.shift(DOWN, 32 - player[1], new_screen)

                # Store transition in the replay buffer.
                screen = new_screen

                episode_rewards[-1] += rew
                reward = episode_rewards[-1]
                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("Episode reward", mean_100ep_reward)


if __name__ == '__main__':
    main()
