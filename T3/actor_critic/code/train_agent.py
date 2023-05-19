import gym
import time
import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt

from actor_critic import ActorCriticAgent

def perform_single_rollout(env, agent, render=False):
    
    # Modify this function to return a tuple of numpy arrays containing:
    # (np.array(obs_t), np.array(acs_t), np.arraw(rew_t), np.array(obs_t1), np.array(done_t))
    # np.array(obs_t)   -> shape: (time_steps, nb_obs)
    # np.array(obs_t1)  -> shape: (time_steps, nb_obs)
    # np.array(acs_t)   -> shape: (time_steps, nb_acs) if actions are continuous, (time_steps,) if actions are discrete
    # np.array(rew_t)   -> shape: (time_steps,)
    # np.array(done_t)  -> shape: (time_steps,)

    ob_t = env.reset()
    
    obs_list = []
    action_list = []
    reward_list = []
    obs1_list = []
    done_list = []

    done = False
    episode_reward = 0
    nb_steps = 0

    while not done:

        if render:
            env.render()
            time.sleep(1. / 60)

        action = agent.select_action(ob_t)

        try:    
            ob_t1, reward, done, _ = env.step(action)

        except:

            ob_t1, reward, done, _ = env.step(action.item())
        
        #ob_t1, reward, done, _ = env.step(action)


        obs_list.append(ob_t)
        action_list.append(action)
        reward_list.append(reward)
        obs1_list.append(ob_t1)
        done_list.append(done)

        ob_t = np.squeeze(ob_t1)
        episode_reward += reward
        
        nb_steps += 1

        if done:
            #print(f"Largo del episodio {nb_steps}")
            obs_array = np.array(obs_list)
            action_array = np.array(action_list)
            reward_array = np.array(reward_list)
            obs1_array = np.array(obs1_list)
            done_array = np.array(done_list)

            return obs_array, action_array, reward_array,obs1_array,done_array


def sample_rollouts(env, agent, training_iter, min_batch_steps):

    sampled_rollouts = []
    total_nb_steps = 0
    episode_nb = 0
    
    while total_nb_steps < min_batch_steps:

        episode_nb += 1
        #render = training_iter%10 == 0 and len(sampled_rollouts) == 0
        render=False
        # Use perform_single_rollout to get data 
        # Uncomment once perform_single_rollout works.
        # Return sampled_rollouts
        
        sample_rollout = perform_single_rollout(env, agent, render=render)
        total_nb_steps += len(sample_rollout[0])

        sampled_rollouts.append(sample_rollout)
        
        
    return sampled_rollouts


def train_agent(env, agent, training_iterations, min_batch_steps, nb_critic_updates,id_str='exp'):

    tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec = [], [], [], []
    _, (axes) = plt.subplots(1, 2, figsize=(12,4))

    for tr_iter in range(training_iterations + 1):

        # Sample rollouts using sample_rollouts
        sampled_rollouts = sample_rollouts(env, agent, training_iterations, min_batch_steps)

        # performed_batch_steps >= min_batch_steps
        # Parse samples into the following arrays:

        sampled_obs_t  = np.concatenate([sampled_rollouts[i][0] for i in range(len(sampled_rollouts))])   # sampled_obs_t:  Numpy array, shape: (performed_batch_steps, nb_observations)
        sampled_acs_t  = np.concatenate([sampled_rollouts[i][1] for i in range(len(sampled_rollouts))])   # sampled_acs:    Numpy array, shape: (performed_batch_steps, nb_actions) if actions are continuous, 
                                #                                     (performed_batch_steps,)            if actions are discrete
        sampled_rew_t  = np.concatenate([sampled_rollouts[i][2] for i in range(len(sampled_rollouts))])   # sampled_rew_t:  Numpy array, shape: (performed_batch_steps,)
        sampled_obs_t1 = np.concatenate([sampled_rollouts[i][3] for i in range(len(sampled_rollouts))])   # sampled_obs_t1: Numpy array, shape: (performed_batch_steps, nb_observations)
        sampled_done_t = np.concatenate([sampled_rollouts[i][4] for i in range(len(sampled_rollouts))])   # sampled_done_t: Numpy array, shape: (performed_batch_steps,)


        # performance metrics
        update_performance_metrics(tr_iter, sampled_rollouts, axes, tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec)

        for _ in range(nb_critic_updates):
            agent.update_critic(sampled_obs_t, sampled_rew_t, sampled_obs_t1, sampled_done_t)
        
        agent.update_actor(sampled_obs_t, sampled_acs_t, sampled_rew_t, sampled_obs_t1, sampled_done_t)

    save_metrics(tr_iters_vec, avg_reward_vec, std_reward_vec,id_str)


def update_performance_metrics(tr_iter, sampled_rollouts, axes, tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec):

    raw_returns     = np.array([np.sum(rollout[2]) for rollout in sampled_rollouts])
    rollout_steps   = np.array([len(rollout[2]) for rollout in sampled_rollouts])

    avg_return = np.average(raw_returns)
    max_episode_return = np.max(raw_returns)
    min_episode_return = np.min(raw_returns)
    std_return = np.std(raw_returns)
    avg_steps = np.average(rollout_steps)

    # logs 
    print('-' * 32)
    print('%20s : %5d'   % ('Training iter'     ,(tr_iter)              ))
    print('-' * 32)
    print('%20s : %5.3g' % ('Max episode return', max_episode_return    ))
    print('%20s : %5.3g' % ('Min episode return', min_episode_return    ))
    print('%20s : %5.3g' % ('Return avg'        , avg_return            ))
    print('%20s : %5.3g' % ('Return std'        , std_return            ))
    print('%20s : %5.3g' % ('Steps avg'         , avg_steps             ))

    avg_reward_vec.append(avg_return)
    std_reward_vec.append(std_return)

    avg_steps_vec.append(avg_steps)

    tr_iters_vec.append(tr_iter)

    #plot_performance_metrics(axes, 
    #                        tr_iters_vec, 
    #                        avg_reward_vec, 
    #                        std_reward_vec, 
    #                        avg_steps_vec)


def plot_performance_metrics(axes, tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec):
    ax1, ax2 = axes
    
    [ax.cla() for ax in axes]
    ax1.errorbar(tr_iters_vec, avg_reward_vec, yerr=std_reward_vec, marker='.',color='C0')
    ax1.set_ylabel('Avg Reward')
    ax2.plot(tr_iters_vec, avg_steps_vec, marker='.',color='C1')
    ax2.set_ylabel('Avg Steps')

    [ax.grid('on') for ax in axes]
    [ax.set_xlabel('training iteration') for ax in axes]
    plt.pause(0.05)


def save_metrics(tr_iters_vec, avg_reward_vec, std_reward_vec,id_str):
    with open(id_str+'.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['steps', 'avg_reward', 'std_reward'])
        for i in range(len(tr_iters_vec)):
            csv_writer.writerow([tr_iters_vec[i], avg_reward_vec[i], std_reward_vec[i]])


if __name__ == '__main__':

    '''
    env = gym.make('Pendulum-v1')
    #env = gym.make('CartPole-v1')
    #env = gym.make('Acrobot-v1')
    dim_states = env.observation_space.shape[0]

    continuous_control = isinstance(env.action_space, gym.spaces.Box)

    dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

    actor_critic_agent = ActorCriticAgent(dim_states=dim_states,
                                          dim_actions=dim_actions,
                                          actor_lr=0.001,
                                          critic_lr=0.001,
                                          gamma=0.99,
                                          continuous_control=continuous_control)

    train_agent(env=env, 
                agent=actor_critic_agent, 
                training_iterations=200, 
                min_batch_steps=5000,
                nb_critic_updates=100)
    '''
    '''
    ################ Experimentos CartPole #############################
    
    actor_lr=0.001
    critic_lr=0.001
    gamma=0.99
    training_iterations=200

    exp_11={"name":"exp_11", "batch_size":500, "nb_critic_updates":1}
    exp_21={"name":"exp_21", "batch_size":500, "nb_critic_updates":10}
    exp_31={"name":"exp_31", "batch_size":500, "nb_critic_updates":100}
   
    exp_12={"name":"exp_12", "batch_size":5000,"nb_critic_updates":1}
    exp_22={"name":"exp_22", "batch_size":5000,"nb_critic_updates":10}
    exp_32={"name":"exp_32", "batch_size":5000,"nb_critic_updates":100}
   
    experimentos=[exp_11,
                  exp_21,
                   exp_31,#exp_12,exp_22,exp_32
                   ]

    # Iteramos sobre cada conjunto de experimentos
    for exp in experimentos:

        # Parametros según experimento
        batch_size=exp["batch_size"]
        nb_critic_updates=exp["nb_critic_updates"]
        
        # Cada experimento se ejecuta 3 veces
        for num_exp in range(1,4):
            
            # Id identificador del experimento
            id_str=exp["name"]+'_'+str(num_exp)+'_'+"CartPole"

            # env
            env = gym.make('CartPole-v1')

            dim_states = env.observation_space.shape[0]

            continuous_control = isinstance(env.action_space, gym.spaces.Box)

            dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

            actor_critic_agent = ActorCriticAgent(dim_states=dim_states,
                                          dim_actions=dim_actions,
                                          actor_lr=actor_lr,
                                          critic_lr=critic_lr,
                                          gamma=gamma,
                                          continuous_control=continuous_control)

            train_agent(env=env, 
                agent=actor_critic_agent, 
                training_iterations=training_iterations, 
                min_batch_steps=batch_size,
                nb_critic_updates=nb_critic_updates,
                id_str=id_str)
    '''     
    '''
    ################ Experimentos Acrobot #############################
    
    actor_lr=0.001
    critic_lr=0.001
    gamma=0.99
    training_iterations=200

    exp_11={"name":"exp_11", "batch_size":500, "nb_critic_updates":1}
    exp_21={"name":"exp_21", "batch_size":500, "nb_critic_updates":10}
    exp_31={"name":"exp_31", "batch_size":500, "nb_critic_updates":100}
   
    exp_12={"name":"exp_12", "batch_size":5000,"nb_critic_updates":1}
    exp_22={"name":"exp_22", "batch_size":5000,"nb_critic_updates":10}
    exp_32={"name":"exp_32", "batch_size":5000,"nb_critic_updates":100}
   
    experimentos=[exp_11,
                  exp_21,
                   exp_31,exp_12,exp_22,exp_32]

    # Iteramos sobre cada conjunto de experimentos
    for exp in experimentos:

        # Parametros según experimento
        batch_size=exp["batch_size"]
        nb_critic_updates=exp["nb_critic_updates"]
        
        # Cada experimento se ejecuta 3 veces
        for num_exp in range(1,4):
            
            # Id identificador del experimento
            id_str=exp["name"]+'_'+str(num_exp)+'_'+"Acrobot"

            # env
            env = gym.make('Acrobot-v1')

            dim_states = env.observation_space.shape[0]

            continuous_control = isinstance(env.action_space, gym.spaces.Box)

            dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

            actor_critic_agent = ActorCriticAgent(dim_states=dim_states,
                                          dim_actions=dim_actions,
                                          actor_lr=actor_lr,
                                          critic_lr=critic_lr,
                                          gamma=gamma,
                                          continuous_control=continuous_control)

            train_agent(env=env, 
                agent=actor_critic_agent, 
                training_iterations=training_iterations, 
                min_batch_steps=batch_size,
                nb_critic_updates=nb_critic_updates,
                id_str=id_str)
    '''
    
    ################ Experimentos Pendulum 1 #############################
    
    actor_lr=0.001
    critic_lr=0.001
    #critic_lr=0.01
    gamma=0.99
    training_iterations=2000


    exp_12={"name":"exp_12", "batch_size":5000,"nb_critic_updates":1}
    exp_22={"name":"exp_22", "batch_size":5000,"nb_critic_updates":10}
    exp_32={"name":"exp_32", "batch_size":5000,"nb_critic_updates":100}
   
    experimentos=[#exp_12,exp_22,
    exp_32]

    # Iteramos sobre cada conjunto de experimentos
    for exp in experimentos:
        print(exp)

        # Parametros según experimento
        batch_size=exp["batch_size"]
        nb_critic_updates=exp["nb_critic_updates"]
        
        # Cada experimento se ejecuta 3 veces
        for num_exp in range(1,4):
            
            # Id identificador del experimento
            id_str=exp["name"]+'_'+str(num_exp)+'_'+"Pendulum_1"

            # env
            env = gym.make('Pendulum-v1')

            dim_states = env.observation_space.shape[0]

            continuous_control = isinstance(env.action_space, gym.spaces.Box)

            dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

            actor_critic_agent = ActorCriticAgent(dim_states=dim_states,
                                          dim_actions=dim_actions,
                                          actor_lr=actor_lr,
                                          critic_lr=critic_lr,
                                          gamma=gamma,
                                          continuous_control=continuous_control)

            train_agent(env=env, 
                agent=actor_critic_agent, 
                training_iterations=training_iterations, 
                min_batch_steps=batch_size,
                nb_critic_updates=nb_critic_updates,
                id_str=id_str)


    ################ Experimentos Pendulum 2 #############################
    
    actor_lr=0.001
    #critic_lr=0.001
    critic_lr=0.01
    gamma=0.99
    training_iterations=2000


    exp_12={"name":"exp_12", "batch_size":5000,"nb_critic_updates":1}
    exp_22={"name":"exp_22", "batch_size":5000,"nb_critic_updates":10}
    exp_32={"name":"exp_32", "batch_size":5000,"nb_critic_updates":100}
   
    experimentos=[exp_12,exp_22,exp_32]

    # Iteramos sobre cada conjunto de experimentos
    for exp in experimentos:
        print(exp)
        # Parametros según experimento
        batch_size=exp["batch_size"]
        nb_critic_updates=exp["nb_critic_updates"]
        
        # Cada experimento se ejecuta 3 veces
        for num_exp in range(1,4):
            
            # Id identificador del experimento
            id_str=exp["name"]+'_'+str(num_exp)+'_'+"Pendulum_2"
            print(id_str)
            # env
            env = gym.make('Pendulum-v1')

            dim_states = env.observation_space.shape[0]

            continuous_control = isinstance(env.action_space, gym.spaces.Box)

            dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

            actor_critic_agent = ActorCriticAgent(dim_states=dim_states,
                                          dim_actions=dim_actions,
                                          actor_lr=actor_lr,
                                          critic_lr=critic_lr,
                                          gamma=gamma,
                                          continuous_control=continuous_control)

            train_agent(env=env, 
                agent=actor_critic_agent, 
                training_iterations=training_iterations, 
                min_batch_steps=batch_size,
                nb_critic_updates=nb_critic_updates,
                id_str=id_str) 