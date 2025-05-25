import os  
import random  
import warnings  
import numpy as np  
import tensorflow as tf  
from collections import deque  
import gym

# Suppress warnings and set environment variables
warnings.simplefilter('ignore')  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['PYTHONHASHSEED'] = '0'

# Disable eager execution
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Set random seeds for reproducibility
random.seed(100)
np.random.seed(100)
tf.random.set_seed(100)

class DQLAgent:
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.1
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.9
        self.learning_rate = 0.0001
        self.trewards = list()
        self.max_treward = 0
        
        # Create environment
        self.env = gym.make('CartPole-v1')
        
        # Create TF graph
        self._build_model()
        
        # Create TF session
        self.session = tf.compat.v1.Session()
        self.session.run(tf.compat.v1.global_variables_initializer())
        
    def _build_model(self):
        # Define input placeholders
        self.state_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name='state_input')
        self.target_q = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name='target_q_values')
        self.actions = tf.compat.v1.placeholder(tf.int32, shape=[None], name='actions')
        
        # Define network architecture
        dense1 = tf.compat.v1.layers.dense(self.state_input, 24, activation=tf.nn.relu, name='dense1')
        dense2 = tf.compat.v1.layers.dense(dense1, 24, activation=tf.nn.relu, name='dense2')
        self.q_values = tf.compat.v1.layers.dense(dense2, 2, name='q_values')
        
        # Define training operations
        self.action_masks = tf.one_hot(self.actions, 2, name='action_masks')
        self.q_values_for_actions = tf.reduce_sum(self.q_values * self.action_masks, axis=1)
        
        # Loss function and optimizer
        self.target_q_for_actions = tf.reduce_sum(self.target_q * self.action_masks, axis=1)
        self.loss = tf.reduce_mean(tf.square(self.target_q_for_actions - self.q_values_for_actions))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
    
    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Get Q values from the model
        q_values = self.session.run(self.q_values, feed_dict={self.state_input: state})
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Extract data from batch
        states = np.vstack([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        next_states = np.vstack([exp[2] for exp in batch])
        rewards = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=np.bool_)
        
        # Get current Q values for all states
        current_q_values = self.session.run(self.q_values, feed_dict={self.state_input: states})
        
        # Get Q values for next states
        next_q_values = self.session.run(self.q_values, feed_dict={self.state_input: next_states})
        
        # Prepare target Q values
        target_q_values = current_q_values.copy()
        
        # Update target Q values based on Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        _, loss_value = self.session.run(
            [self.train_op, self.loss],
            feed_dict={
                self.state_input: states,
                self.target_q: target_q_values,
                self.actions: actions
            }
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss_value

    def learn(self, episodes):
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, 4])
            
            for t in range(1, 5000):
                action = self.act(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                
                # Store experience in memory
                self.memory.append((state, action, next_state, reward, done))
                
                state = next_state
                
                if done or trunc:
                    self.trewards.append(t)
                    self.max_treward = max(self.max_treward, t)
                    templ = f'episode={e:4d} | treward={t:4d}'
                    templ += f' | max={self.max_treward:4d}'
                    print(templ, end='\r')
                    break
            
            # Perform experience replay after each episode
            if len(self.memory) >= self.batch_size:
                self.replay()
                
        print("\nTraining completed!")
        
    def test(self, episodes):
        print("\nTesting agent performance:")
        total_reward = 0
        
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, 4])
            episode_reward = 0
            
            for t in range(1, 5001):
                # Get action using the model
                q_values = self.session.run(self.q_values, feed_dict={self.state_input: state})
                action = np.argmax(q_values[0])
                
                state, reward, done, trunc, _ = self.env.step(action)
                state = np.reshape(state, [1, 4])
                episode_reward += reward
                
                if done or trunc:
                    print(f"Episode {e}: {t} steps", end=' | ')
                    total_reward += episode_reward
                    break
                    
        print(f"\nAverage reward over {episodes} test episodes: {total_reward/episodes:.2f}")

# Create the agent
agent = DQLAgent()

# Train the agent
agent.learn(1500)

# Test the agent after training
agent.test(10)