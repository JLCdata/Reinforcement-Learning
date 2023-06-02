import gym
import time

import numpy as np
import matplotlib.pyplot as plt

class MCarExpertPolicy:

    def __init__(self):
        self._policy =  ('10111022201222220222221112002220022102221210221010021210210100200011121222200000012212221200000'+
                         '00022222110000000022222220001010022022100201001021022220200000000022100121101001222202201001002'+
                         '22200222010002221001111010112220001002102002220200200112202220212111222122220012002210201111112')


    def select_action(self, observation):
        return int(self._policy[int(np.round(observation[0] * 10. + 12.)) * 15 + int(np.round(observation[1] * 100. + 7.))])


    def plot_policy(self):
        policy_vec = np.reshape(list(map(int, list(self._policy))), (15, 19))
        plt.imshow(policy_vec, cmap='gray')
        plt.xticks(range(0, 19, 2), np.array(range(-12, 7, 2)) * 0.1)
        plt.yticks(range(0, 15, 2), np.array(range(-7, 8, 2)) * 0.01)
        plt.xlabel('position')
        plt.ylabel('velocity')
        plt.show()


def test_agent(env, agent, nb_episodes=30, render=True):

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

            ob_t = ob_t1
            episode_reward += reward
            
            nb_steps += 1

            if done:
                if ob_t[0] > 0.5:
                    success_rate += 1.
                avg_steps += nb_steps
                ep_rewards.append(episode_reward)
                print('Evaluation episode %3d | Steps: %4d | Reward: %4d | Success: %r' % (episode + 1, nb_steps, episode_reward, ob_t[0] > 0.5))
    
    ep_rewards = np.array(ep_rewards)
    avg_reward = np.average(ep_rewards)
    std_reward = np.std(ep_rewards)
    success_rate /= nb_episodes
    avg_steps /= nb_episodes
    print('Average Reward: %.2f, Reward Deviation: %.2f | Average Steps: %.2f, Success Rate: %.2f' % (avg_reward, std_reward, avg_steps, success_rate))


if __name__ == '__main__':

    env = gym.make('MountainCar-v0')
    expert_agent = MCarExpertPolicy()
    test_agent(env, expert_agent)
    expert_agent.plot_policy()
