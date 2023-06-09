import time
import torch
import datetime
import gym
import csv

import numpy as np

import matplotlib.pyplot as plt

from mbrl import MBRLAgent


def cartpole_reward(observation_batch, action_batch):
    # obs = (x, x', theta, theta')
    # rew = cos(theta) - 0.01 * x^2

    x=observation_batch[:,0]
    theta = observation_batch[:, 2]
    reward = np.cos(theta) - 0.01 * x**2
    
    return reward

def pendulum_reward(observation_batch, action_batch):
    # obs = (cos(theta), sin(theta), theta')
    # rew = - theta^2 - 0.1 * (theta')^2 - 0.001 * a^2
    
    observation_batch[:,0]=np.clip(observation_batch[:,0],-1,1)
    observation_batch[:,1]=np.clip(observation_batch[:,1],-1,1)
    #observation_batch[:,2]=np.clip(observation_batch[:,2],-8,8)
    # np.arcos(observation_batch[:, 1])
    theta=np.arctan(observation_batch[:, 1]/observation_batch[:, 0])
    # Norm 
    theta_dot = observation_batch[:, 2]
    reward = -theta**2 - 0.1 * theta_dot**2 - 0.001 * action_batch**2
    
    return reward


def train_agent(env, eval_env, agent, nb_training_steps, nb_data_collection_steps,
                nb_epochs_for_model_training, nb_steps_between_model_updates, render=False):

    tr_steps_vec, avg_reward_vec, std_reward_vec = [], [], []
    _, (axes) = plt.subplots(1, 1, figsize=(12,4))

    ob_t = env.reset()
    done = False
    episode_nb = 0
    episode_reward = 0
    episode_steps = 0

    update_performance_metrics(agent, eval_env, 0, axes, tr_steps_vec, 
                               avg_reward_vec, std_reward_vec)

    print("Collecting data to train the model")
    for model_tr_step in range(nb_data_collection_steps):

        action = agent.select_action(ob_t, random=True)

        ob_t1, reward, done, _ = env.step(action)

        agent.store_transition(ob_t, action, ob_t1)

        ob_t = ob_t1

        if render:
            env.render()
            time.sleep(1 / 60.)

        episode_reward += reward
        episode_steps += 1

        if done:
            episode_nb += 1
            ob_t = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
    
    print("Done collecting data, now training")
    for _ in range(nb_epochs_for_model_training):
        agent.update_model()

    
    # Part II only
    for tr_step in range(nb_training_steps):

        if (tr_step + 1) % (nb_training_steps / 5) == 0:
            update_performance_metrics(agent, eval_env, tr_step + 1, axes, tr_steps_vec, 
                                        avg_reward_vec, std_reward_vec)


        if (tr_step + 1) % nb_steps_between_model_updates == 0:
            print("Updating model")
            for _ in range(nb_epochs_for_model_training):
                agent.update_model()

        action = agent.select_action(ob_t)

        ob_t1, reward, done, _ = env.step(action)

        agent.store_transition(ob_t, action, ob_t1)

        ob_t = ob_t1

        if render:
            env.render()
            time.sleep(1 / 60.)

        episode_reward += reward
        episode_steps += 1

        if done:
            print('Global training step %5d | Training episode %5d | Steps: %4d | Reward: %4.2f' % \
                        (tr_step + 1, episode_nb + 1, episode_steps, episode_reward))

            episode_nb += 1
            ob_t = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0

    save_metrics(agent, tr_steps_vec, avg_reward_vec, std_reward_vec)
    

def update_performance_metrics(agent, eval_env, training_step, axes, tr_steps_vec, avg_reward_vec, std_reward_vec):

    avg_reward, std_reward = test_agent(eval_env, agent)

    tr_steps_vec.append(training_step)
    avg_reward_vec.append(avg_reward)
    std_reward_vec.append(std_reward)

    plot_performance_metrics(axes, 
                             tr_steps_vec, 
                             avg_reward_vec, 
                             std_reward_vec)


def save_metrics(agent, tr_steps_vec, avg_reward_vec, std_reward_vec):
    with open('metrics'+datetime.datetime.now().strftime('%H-%M-%S')+'.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['steps', 'avg_reward', 'std_reward'])
            for i in range(len(tr_steps_vec)):
                csv_writer.writerow([tr_steps_vec[i], avg_reward_vec[i], std_reward_vec[i]])

             
def test_agent(env, agent, nb_episodes=30, render=False):

    ep_rewards = []
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
                avg_steps += nb_steps
                ep_rewards.append(episode_reward)
                print('Evaluation episode %3d | Steps: %4d | Reward: %4.2f' % (episode + 1, nb_steps, episode_reward))
    
    ep_rewards = np.array(ep_rewards)
    avg_reward = np.average(ep_rewards)
    std_reward = np.std(ep_rewards)
    avg_steps /= nb_episodes
    print('Average Reward: %.2f, Reward Deviation: %.2f | Average Steps: %.2f' % (avg_reward, std_reward, avg_steps))

    return avg_reward, std_reward


def plot_performance_metrics(axes, tr_steps_vec, avg_reward_vec, std_reward_vec):
    axes.cla()
    axes.errorbar(tr_steps_vec, avg_reward_vec, yerr=std_reward_vec, marker='.',color='C0')
    axes.set_ylabel('Avg Reward')

    axes.grid('on') 
    axes.set_xlabel('Training step')
    plt.pause(0.05)


if __name__ == '__main__':
    
    env_name = 'CartPole-v1'
    #env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    dim_states = env.observation_space.shape[0]
    continuous_control = isinstance(env.action_space, gym.spaces.Box)
    dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n
    reward_function = cartpole_reward if env_name == 'CartPole-v1' else pendulum_reward

    mbrl_agent = MBRLAgent(dim_states=dim_states, 
                           dim_actions=dim_actions,
                           continuous_control=continuous_control,
                           model_lr=0.001,
                           buffer_size=30000,
                           batch_size=256,
                           planning_horizon=30,
                           nb_trajectories=100, 
                           reward_function=reward_function)

    train_agent(env=env, 
                eval_env=eval_env,
                agent=mbrl_agent,
                nb_data_collection_steps=10000,
                nb_steps_between_model_updates=1000,
                nb_epochs_for_model_training=10,
                nb_training_steps=30000)
    

    ''''
    ################ Experimentos CartPole #############################
    exp_11={"name":"exp_11", "batch_size":500, "use_baseline":False,"reward_to_go":False}
    exp_21={"name":"exp_21", "batch_size":500, "use_baseline":False,"reward_to_go":True}
    exp_31={"name":"exp_31", "batch_size":500, "use_baseline":True,"reward_to_go":False}
    exp_41={"name":"exp_41", "batch_size":500, "use_baseline":True,"reward_to_go":True}    

    exp_12={"name":"exp_12", "batch_size":5000, "use_baseline":False,"reward_to_go":False}
    exp_22={"name":"exp_22", "batch_size":5000, "use_baseline":False,"reward_to_go":True}
    exp_32={"name":"exp_32", "batch_size":5000, "use_baseline":True,"reward_to_go":False}
    exp_42={"name":"exp_42", "batch_size":5000, "use_baseline":True,"reward_to_go":True}

    experimentos=[exp_11,exp_21,exp_31,exp_41,exp_12,exp_22,exp_32,exp_42]
    
    # Número de steps
    training_iterations=100
    lr=0.005
    gamma=0.99

    # Iteramos sobre cada conjunto de experimentos
    for exp in experimentos:

        # Parametros según experimento
        batch_size=exp["batch_size"]
        use_baseline=exp["use_baseline"]
        reward_to_go=exp["reward_to_go"]

        # Cada experimento se ejecuta 3 veces
        for num_exp in range(1,4):
            
            # Id identificador del experimento
            id_str=exp["name"]+'_'+str(num_exp)+'_'+"CartPole"

            # env
            env = gym.make('CartPole-v1')

            dim_states = env.observation_space.shape[0]

            continuous_control = isinstance(env.action_space, gym.spaces.Box)

            dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

            policy_gradients_agent = PolicyGradients(dim_states=dim_states, 
                                                    dim_actions=dim_actions, 
                                                    lr=lr,
                                                    gamma=gamma,
                                                    continuous_control=continuous_control,
                                                    reward_to_go=reward_to_go,
                                                    use_baseline=use_baseline)

            train_pg_agent(env=env, 
                        agent=policy_gradients_agent, 
                        training_iterations=training_iterations,
                        min_batch_steps=batch_size,
                        id_str=id_str)
            
    '''
    '''
    ################ Experimentos Pendulum #############################
    exp_12={"name":"exp_12", "batch_size":5000, "use_baseline":False,"reward_to_go":False}
    exp_22={"name":"exp_22", "batch_size":5000, "use_baseline":False,"reward_to_go":True}
    exp_32={"name":"exp_32", "batch_size":5000, "use_baseline":True,"reward_to_go":False}
    exp_42={"name":"exp_42", "batch_size":5000, "use_baseline":True,"reward_to_go":True}

    experimentos=[exp_12,exp_22,exp_32,exp_42]
    
    # Número de steps
    training_iterations=1000
    lr=0.005
    gamma=0.99

    # Iteramos sobre cada conjunto de experimentos
    for exp in experimentos:

        # Parametros según experimento
        batch_size=exp["batch_size"]
        use_baseline=exp["use_baseline"]
        reward_to_go=exp["reward_to_go"]

        # Cada experimento se ejecuta 3 veces
        for num_exp in range(1,4):
            
            # Id identificador del experimento
            id_str=exp["name"]+'_'+str(num_exp)+'_'+"Pendulum"

            # env
            env = gym.make('Pendulum-v1')
            
            dim_states = env.observation_space.shape[0]

            continuous_control = isinstance(env.action_space, gym.spaces.Box)

            dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

            policy_gradients_agent = PolicyGradients(dim_states=dim_states, 
                                                    dim_actions=dim_actions, 
                                                    lr=lr,
                                                    gamma=gamma,
                                                    continuous_control=continuous_control,
                                                    reward_to_go=reward_to_go,
                                                    use_baseline=use_baseline)

            train_pg_agent(env=env, 
                        agent=policy_gradients_agent, 
                        training_iterations=training_iterations,
                        min_batch_steps=batch_size,
                        id_str=id_str)
    '''