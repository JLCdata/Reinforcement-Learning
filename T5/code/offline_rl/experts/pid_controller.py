import gym
import time
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, dt=0.02):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._dt = dt
        
        # P1-1
        # Define aux variables (if any)
        self.last_error = 0
        self.integral = 0
        
    def select_action(self, observation):
        # P1-1
        # Set point (do not change)
        error = observation[2]

        # PID control
        # Code the PID control law
        proportional = self._kp * error
        self.integral += self._ki * error * self._dt
        derivative = self._kd * (error - self.last_error) / self._dt
        ctrl = proportional + self.integral + derivative

        # Update error
        self.last_error = error

        return 0 if ctrl < 0 else 1
    
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
                if nb_steps == 200:
                    success_rate += 1.
                avg_steps += nb_steps
                ep_rewards.append(episode_reward)
                print('Evaluation episode %3d | Steps: %4d | Reward: %4d | Success: %r' % (episode + 1, nb_steps, episode_reward, nb_steps == 200))
    
    ep_rewards = np.array(ep_rewards)
    avg_reward = np.average(ep_rewards)
    std_reward = np.std(ep_rewards)
    success_rate /= nb_episodes
    avg_steps /= nb_episodes
    print('Average Reward: %.2f| Reward Deviation: %.2f | Average Steps: %.2f| Success Rate: %.2f' % (avg_reward, std_reward, avg_steps, success_rate))


# P1-2, P1-3
if __name__ == '__main__':

    # Exploración de parámetros:
    # do not change dt = 0.02
    env = gym.make('CartPole-v0')
    print(0.1, 0.3, 0.7)
    pid_agent = PIDController(0.1, 0.3, 0.7, 0.02)
    test_agent(env, pid_agent)

    print(0.5, 0.3, 0.1)
    env = gym.make('CartPole-v0')
    pid_agent = PIDController(0.5, 0.3, 0.1, 0.02)
    test_agent(env, pid_agent)

    print(1, 1.5, 3)
    env = gym.make('CartPole-v0')
    pid_agent = PIDController(1, 1.5, 3, 0.02)
    test_agent(env, pid_agent)

    print(3, 2, 1.5)
    env = gym.make('CartPole-v0')
    pid_agent = PIDController(3, 2, 1.5, 0.02)
    test_agent(env, pid_agent)

    print(2, 3, 4)
    env = gym.make('CartPole-v0')
    pid_agent = PIDController(2, 3, 4, 0.02)
    test_agent(env, pid_agent)

    print(4, 2, 3)
    env = gym.make('CartPole-v0')
    pid_agent = PIDController(4, 2, 3, 0.02)
    test_agent(env, pid_agent)

    print(5, 5, 5)
    env = gym.make('CartPole-v0')
    pid_agent = PIDController(5, 5, 5, 0.02)
    test_agent(env, pid_agent)
