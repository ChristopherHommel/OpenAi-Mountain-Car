import gym
import numpy as np

env = gym.make('MountainCar-v0')

LEARNING_RATE = 0.25
DISCOUNT = 0.95
EPISODES = 50000
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

DISCRETE_OS_SIZE = [50] * len(env.observation_space.high)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def exploration_exploitation_decay(epsilon):
    epsilon_decay_value = epsilon /(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    return epsilon_decay_value
    

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / ((env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE)
    return tuple(discrete_state.astype(np.int32))

def calculate_new_q_value(discrete_state, reward, action):
    max_future_q = np.max(q_table[discrete_state])

    current_q = q_table[discrete_state + (action,)]

    #1-alpha -> learning_rate = alpha
    #Get the current q value and add alpha
    #calculate the learned value from reward + discount factor * max value in the q table
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

    return new_q

def get_action(epsilon, discrete_state):
    if np.random.random() > epsilon:
        action = np.argmax(q_table[discrete_state])
    else:
        action = np.random.randint(0, env.action_space.n)
    
    return action

def state(episode, action):
    print(f'Episodes - {episode} , Action - {action}')
    
def main():
    epsilon = 1
    
    for episode in range(EPISODES):
        discrete_state = get_discrete_state(env.reset())
        done = False

        while not done:
            action = get_action(epsilon, discrete_state)
            
            new_state, reward, done, info = env.step(action)

            new_discrete_state = get_discrete_state(new_state)

            if episode % 2500 == 0:
                env.render()
                
            if not done:
                #update the q table
                q_table[discrete_state + (action,)] = calculate_new_q_value(new_discrete_state, reward, action)
                
            #end goal position = 0.45
            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state
            
            #state(episode, action)
            
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon_decay_value = exploration_exploitation_decay(epsilon)
            epsilon -= epsilon_decay_value
                     
    env.close()

if __name__ == '__main__':
    main()
    

    
