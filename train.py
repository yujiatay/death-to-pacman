import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import pandas as pd
import os
import math

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=200000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=2, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="PlaceHolder", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./save_files/", help="directory in which training state and "
                                                                              "model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes " \
                                                                     "are "
                                                                    "completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are "
                                                                 "loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data"
                                                                                        " is saved")
    parser.add_argument("--plots-dir", type=str, default="./save_files/", help="directory where plot data is saved")

    #Newly added arguments
    parser.add_argument("--load", action="store_true", default=False) #only load if this is true. So we can display without loading
    parser.add_argument("--load_episode",type = int, default=0)
    parser.add_argument("--layout", type=str, default="smallClassic") #decide the layout to train
    parser.add_argument("--obs_type", type=str, default="full_obs")  # full_obs or partial_obs
    parser.add_argument("--partial_obs_range", type=int, default=3)  # 3x3,5x5,7x7 ...
    parser.add_argument("--shared_obs", action="store_true", default= False)  # pacman and ghost same observation?
    parser.add_argument("--timeStepObs", action="store_true", default= False)  # Do we want 2 time steps?
    parser.add_argument("--astarSearch", action="store_true", default= False)  # Do we want negative reward for dist
    parser.add_argument("--astarAlpha", type=int, default= 1)  # How much do we penalize them

    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    n = int(input.shape[1])
    m = num_outputs
    first_layer = int((math.sqrt((m+2)*n)) + 2*(math.sqrt(n/(m+2))))
    second_layer = int(m*(math.sqrt(n/(m+2))))


    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=max(num_units,first_layer), activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=max(num_units,second_layer), activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from pacman.gym_pacman.envs.pacman_env import PacmanEnv

    env = PacmanEnv(arglist.display,
                    arglist.num_adversaries,
                    arglist.max_episode_len,
                    arglist.layout,  # for random, put string "random"
                    arglist.obs_type,
                    arglist.partial_obs_range,
                    arglist.shared_obs,
                    arglist.timeStepObs,
                    arglist.astarSearch,
                    arglist.astarAlpha)
    env.seed(1)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    print("obs_shape_n", obs_shape_n)
    print("action_space", env.action_space)
    for i in range(1): # Pac-Man
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    for i in range(1, env.n): # Ghosts
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    return trainers


def train(arglist):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_n = env.reset()  # so that env.observation_space is initialized so trainers can be initialized
        # Create agent trainers
        num_adversaries = arglist.num_adversaries
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        print("env.observation_space.shape", env.observation_space.shape)
        print(obs_shape_n)
        print("num adversaries: ", num_adversaries, ", env.n (num agents): ", env.n)

        #need to ensure that the trainer is in correct order. pacman in front
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir + ("{}".format(arglist.load_episode))
        if arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)
        if arglist.display and arglist.load:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = [[] for i in range(env.n)]  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver(max_to_keep=None)
        episode_step = 0
        train_step = 0
        total_win =[0]
        final_win = []
        total_lose = [0]
        final_lose = []
        t_start = time.time()
        loss_list ={}
        for i in range(env.n):
            loss_list[i] = [[] for i in range(6)]


        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done, info_n ,win , lose = env.step(action_n)
            episode_step += 1
            terminal = (episode_step >= arglist.max_episode_len)
            # print("obs_n", obs_n)
            # print("new_obs_n", new_obs_n)
            #print("action_n", action_n)
            # print("rew_n",episode_step, rew_n)
            # print("done", done)
            # print("terminal", terminal)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done, terminal)
            obs_n = new_obs_n
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            if done or terminal:
                if arglist.display:
                    env.render()
                obs_n = env.reset()
                episode_step = 0
                if win:
                    total_win[-1] += 1
                if lose:
                    total_lose[-1] += 1
                total_win.append(0)
                total_lose.append(0)
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1
            # if train_step % 1000 == 0:
            #     print(train_step)
            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None

            for agent in trainers:
                agent.preupdate()
            for ind, agent in enumerate(trainers):
                loss = agent.update(trainers, train_step)
                if train_step%10000 == 0 and loss != None:
                    for i in range(len(loss)):
                        loss_list[ind][i].append(loss[i])



            # save model, display training output
            if (terminal or done) and (len(episode_rewards) % arglist.save_rate == 0):
                saving = arglist.save_dir + ("{}".format(0 + len(episode_rewards))) #TODO why append this
                U.save_state(saving, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, number of wins {}, number of lose {}, "
                          "time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards],np.sum(total_win[-arglist.save_rate:]),np.sum(total_lose[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                final_win.append(np.sum(total_win[-arglist.save_rate:]))
                final_lose.append(np.sum(total_lose[-arglist.save_rate:]))

                ep_reward_df = pd.DataFrame(final_ep_rewards)
                ep_ag_reward_df = pd.DataFrame(final_ep_ag_rewards)
                win_df = pd.DataFrame(final_win)
                lose_df = pd.DataFrame(final_lose)
                for i in range(env.n):
                    trainer_loss_df = pd.DataFrame(loss_list[i]).transpose()
                    trainer_loss_df.to_csv(arglist.plots_dir + arglist.exp_name + '_trainer_loss_df_{}.csv'.format(i))

                ep_reward_df.to_csv(arglist.plots_dir + arglist.exp_name + '_rewards.csv')
                ep_ag_reward_df.to_csv(arglist.plots_dir + arglist.exp_name + '_agrewards.csv')
                win_df.to_csv(arglist.plots_dir + arglist.exp_name + '_win_df.csv')
                lose_df.to_csv(arglist.plots_dir + arglist.exp_name + '_lose_df.csv')



                for i,rew in enumerate(agent_rewards):
                    final_ep_ag_rewards[i].append(np.mean(rew[-arglist.save_rate:]))
            # saves final episode reward for plotting training curve later

            if len(episode_rewards) > arglist.num_episodes:
                # rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                # with open(rew_file_name, 'wb') as fp:
                #     pickle.dump(final_ep_rewards, fp)
                # agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                # with open(agrew_file_name, 'wb') as fp:
                #     pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    print(arglist)
    train(arglist)
