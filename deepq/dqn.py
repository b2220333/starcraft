import torch
import os
from absl import flags
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from common.schedules import LinearSchedule
from common import common, logger

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_UNIT_ID = 1

_CONTROL_GROUP_SET = 1
_CONTROL_GROUP_RECALL = 0

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'
FLAGS = flags.FLAGS
EPSILON = 0.9
cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)


def save_checkpoint(state, filename=''):
    """Save checkpoint if a new best is achieved"""
    if os.path.exists(filename):
        os.remove(filename)
    print("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint


def load_checkpoint(model, cuda=False, filename=''):
    print("=> loading checkpoint '{}' ...".format(filename))
    if cuda:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename,
                                map_location=lambda storage,
                                                    loc: storage)
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(filename,
                                                                     checkpoint['epoch']))
    return model, best_accuracy


class Net(nn.Module):
    def __init__(self, num_actions):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 64, 64)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=32,  # n_filters
                kernel_size=8,  # filter size
                stride=4,  # filter movement/step
                padding=4
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (64, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (32, 32, 32)
        )
        self.conv2 = nn.Sequential(  # input shape (32, 32, 32)
            nn.Conv2d(32, 64, 4, 2, 2),  # output shape (32, 32, 32)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 16, 16)
        )
        self.conv3 = nn.Sequential(  # input shape (64, 16, 16)
            nn.Conv2d(64, 64, 3, 1, 2),  # output shape (64, 16, 16)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 8, 8)
        )
        self.action_out = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.action_scores = nn.Linear(256, num_actions)  # fully connected layer, output 10 classes

    # (32, 8, 4), (64, 4, 2), (64, 3, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.action_out(x))
        actions_value = self.action_scores(x)
        return actions_value


class DQN(object):
    def __init__(self, num_actions=3, lr=5e-4, cuda=False):
        self.eval_net, self.target_net = Net(num_actions), Net(num_actions)
        self.cuda = cuda
        if self.cuda:
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
        self.num_actions = num_actions
        self.learn_step_counter = 0  # for target updating
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.SmoothL1Loss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if self.cuda:
            x = x.cuda()
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            if self.cuda:
                action = torch.max(actions_value, 1)[1].cuda().data.numpy()
                action = action[0].cpu()
            else:
                action = torch.max(actions_value, 1)[1].data.numpy()
                action = action[0]
        else:  # random
            action = np.random.randint(0, self.num_actions)
            action = action
        return action

    def load_state_dict(self, state_dict):
        self.eval_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def save_state_dict(self):
        return self.target_net.state_dict()

    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self, obses_t, actions, rewards, obses_tp1, gamma, batch_size):
        b_s = Variable(torch.FloatTensor(obses_t))
        b_a = Variable(torch.LongTensor(actions.astype(int)))
        b_r = Variable(torch.FloatTensor(rewards))
        b_s_ = Variable(torch.FloatTensor(obses_tp1))
        if self.cuda:
            b_s, b_a, b_r, b_s_ = b_s.cuda(), b_a.cuda(), b_r.cuda(), b_s_.cuda()
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.cuda:
            q_eval, q_target = q_eval.cpu(), q_target.cpu()
        return q_eval - q_target


def learn(env,
          num_actions=3,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          num_cpu=16):
    torch.set_num_threads(num_cpu)
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(
            buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(
            prioritized_replay_beta_iters,
            initial_p=prioritized_replay_beta0,
            final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    exploration = LinearSchedule(
        schedule_timesteps=int(exploration_fraction * max_timesteps),
        initial_p=1.0,
        final_p=exploration_final_eps)
    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

    screen = player_relative

    obs, xy_per_marine = common.init(env, obs)

    group_id = 0
    reset = True
    dqn = DQN(num_actions, lr, cuda)

    print('\nCollecting experience...')
    checkpoint_path = 'models/deepq/checkpoint.pth.tar'
    if os.path.exists(checkpoint_path):
        dqn, saved_mean_reward = load_checkpoint(dqn, cuda, filename=checkpoint_path)
    for t in range(max_timesteps):
        # Take action and update exploration to the newest value
        # custom process for DefeatZerglingsAndBanelings
        obs, screen, player = common.select_marine(env, obs)
        # action = act(
        #     np.array(screen)[None], update_eps=update_eps, **kwargs)[0]
        action = dqn.choose_action(np.array(screen)[None])
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
        replay_buffer.add(screen, action, rew, new_screen, float(done))
        screen = new_screen
        episode_rewards[-1] += rew
        reward = episode_rewards[-1]
        if done:
            print("Episode Reward : %s" % episode_rewards[-1])
            obs = env.reset()
            player_relative = obs[0].observation["screen"][
                _PLAYER_RELATIVE]
            screen = player_relative
            group_list = common.init(env, obs)
            # Select all marines first
            # env.step(actions=[sc2_actions.FunctionCall(_SELECT_UNIT, [_SELECT_ALL])])
            episode_rewards.append(0.0)
            reset = True

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer.sample(
                    batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights,
                 batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(
                    batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

            td_errors = dqn.learn(obses_t, actions, rewards, obses_tp1, gamma, batch_size)

            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes,
                                                new_priorities)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            dqn.update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(
                episode_rewards) % print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("reward", reward)
            logger.record_tabular("mean 100 episode reward",
                                  mean_100ep_reward)
            logger.record_tabular("% time spent exploring",
                                  int(100 * exploration.value(t)))
            logger.dump_tabular()

        if (checkpoint_freq is not None and t > learning_starts
                and num_episodes > 100 and t % checkpoint_freq == 0):
            if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                if print_freq is not None:
                    logger.log(
                        "Saving model due to mean reward increase: {} -> {}".format(
                            saved_mean_reward,
                            mean_100ep_reward))
                save_checkpoint({
                    'epoch': t + 1,
                    'state_dict': dqn.save_state_dict(),
                    'best_accuracy': mean_100ep_reward
                }, checkpoint_path)
                saved_mean_reward = mean_100ep_reward
# /output/checkpoint.pth.tar
