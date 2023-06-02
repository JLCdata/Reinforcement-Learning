import csv
import gym
import time

import datetime
import numpy as np
import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer
from conservative_q_learning import ConservativeDeepQNetworkAgent


def perform_rollouts(env, agent, nb_episodes, replay_buffer=None, render=False):

    env_name = env.unwrapped.spec.id
    ep_rewards = []
    success_rate = 0
    avg_steps = 0

    for episode in range(nb_episodes):

        ob_t = env.reset()
        done = False
        episode_reward = 0
        nb_steps = 0

        while not done:

            if render and episode == 0:
                env.render()
                time.sleep(1. / 60)
                
            action = agent.select_action(ob_t)
            
            ob_t1, reward, done, _ = env.step(action)
            
            if replay_buffer is not None:
                replay_buffer.store_transition(ob_t, action, reward, ob_t1, done)

            ob_t = ob_t1
            episode_reward += reward
            
            nb_steps += 1
            if done:
                if env_name == 'CartPole-v0':
                    if nb_steps == 200:
                        success_rate += 1.
                elif env_name == 'MountainCar-v0':
                    if ob_t[0] > 0.5:
                        success_rate += 1.
                else:
                    raise NotImplementedError()

                avg_steps += nb_steps
                ep_rewards.append(episode_reward)
                #print('Evaluation episode %3d | Steps: %4d | Reward: %4d | Success: %r' % (episode + 1, nb_steps, episode_reward, nb_steps == 200 if env_name=='CartPole-v0' else ob_t[0] > 0.5))
    
    ep_rewards = np.array(ep_rewards)
    avg_reward = np.average(ep_rewards)
    std_reward = np.std(ep_rewards)
    success_rate /= nb_episodes
    avg_steps /= nb_episodes

    print('Average Reward: %.2f, Reward Deviation: %.2f | Average Steps: %.2f, Success Rate: %.2f' % (avg_reward, std_reward, avg_steps, success_rate))

    return replay_buffer, avg_reward, std_reward, success_rate


def train_agent(env, agent, expert_agent, replay_buffer, rollout_episodes, training_iters):

    tr_iters_vec, avg_rew_vec, std_rew_vec, sr_vec = [], [], [], []
    _, (axes) = plt.subplots(1, 2, figsize=(12,4))
    
    demonstrations, _, _, _ = perform_rollouts(env, expert_agent, rollout_episodes, replay_buffer, True)
    
    for iter_nb in range(training_iters + 1):
        batch = replay_buffer.sample_transitions()
        agent.update(batch)

        if (iter_nb % 1000) == 0:
            agent.replace_target_network()
            
            # use render = False to speed up the training process
            _, avg_reward, std_reward, success_rate = perform_rollouts(env, agent, 100, None, False)
            tr_iters_vec.append(iter_nb)
            avg_rew_vec.append(avg_reward)
            std_rew_vec.append(std_reward)
            sr_vec.append(success_rate)
            plot_training_metrics(axes, tr_iters_vec, avg_rew_vec, std_rew_vec, sr_vec)
    
    save_metrics(tr_iters_vec, avg_rew_vec, std_rew_vec, sr_vec)


def plot_training_metrics(axes, tr_iters_vec, avg_rew_vec, std_rew_vec, sr_vec):
    ax1, ax2 = axes
    
    [ax.cla() for ax in axes]
    ax1.errorbar(tr_iters_vec, avg_rew_vec, yerr=std_rew_vec, marker='.', color='C0')
    ax1.set_ylabel('Avg. Return')
    ax2.plot(tr_iters_vec, sr_vec, marker='.', color='C1')
    ax2.set_ylabel('Success Rate')

    [ax.grid('on') for ax in axes]
    [ax.set_xlabel('training steps') for ax in axes]
    plt.pause(0.05)


def save_metrics(tr_iters_vec, avg_reward_vec, std_reward_vec, sr_vec):
    with open('metrics'+datetime.datetime.now().strftime('%H:%M:%S')+'.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(['steps', 'avg_reward', 'std_reward'])
        for i in range(len(tr_iters_vec)):
            csv_writer.writerow([tr_iters_vec[i], avg_reward_vec[i], std_reward_vec[i], sr_vec[i]])

        
if __name__ == '__main__':

    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    if env_name == 'CartPole-v0':
        from experts.pid_controller import PIDController
        # Complete using P1
        expert_agent = PIDController(0.0, 0, 0.0, 0.02)
    
    elif env_name == 'MountainCar-v0':
        from experts.mcar_policy import MCarExpertPolicy
        expert_agent = MCarExpertPolicy()

    else: 
        raise NotImplementedError()

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.n
    
    agent = ConservativeDeepQNetworkAgent(dim_states=nb_states, dim_actions=nb_actions, lr=0.01, gamma=0.99, alpha=20.0)
    replay_buffer = ReplayBuffer(dim_states=nb_states, dim_actions=nb_actions, max_size=100000, sample_size=128)

    train_agent(env=env, agent=agent, expert_agent=expert_agent, 
                replay_buffer=replay_buffer, rollout_episodes=200, training_iters=20000)
