# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:41:13 2020

@author: Connor

Using the Radial Basis function Network for some basic Gym tests
"""

#importing libraries
import gym
from gym import wrappers
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler 
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

# Feature Transformer (Inspired by https://github.com/dennybritz/reinforcement-learning)
# Using RBFSampler (uses Monte Carlo)
class FeatureTransformer:
    def __init__(self, env):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)
        
        # used to concatenate feature vectors since RBF uses scale parameter
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma = 5.0, n_components = 500)),
                ("rbf2", RBFSampler(gamma = 2.0, n_components = 500)),
                ("rbf3", RBFSampler(gamma = 1.0, n_components = 500)),
                ("rbf4", RBFSampler(gamma = 0.5, n_components = 500)),
                ])
        featurizer.fit(scaler.transform(observation_examples))
        
        self.scaler = scaler
        self.featurizer = featurizer
        
    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)
    
# SGDRegressor for every action (also used by Deep Q Learning)
class Model:
  def __init__(self, env, feature_transformer, learning_rate):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    for i in range(env.action_space.n):
      model = SGDRegressor(learning_rate=learning_rate)
      model.partial_fit(feature_transformer.transform( [env.reset()] ), [0])
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform([s])
    result = np.stack([m.predict(X) for m in self.models]).T
    assert(len(result.shape) == 2)
    return result

  def update(self, s, a, G):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    # necessary for gradient decent
    self.models[a].partial_fit(X, [G])

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))
  
# Play a round
def play_one(model, env, eps, gamma):
    obs = env.reset()
    done = False
    totalreward = 0
    i = 0
    while not done and i < 10000:
        action = model.sample_action(obs, eps)
        prev_obs = obs
        obs, reward, done, info = env.step(action)
        
        G = reward + gamma*np.max(model.predict(obs)[0])
        model.update(prev_obs, action, G)
        
        totalreward += reward
        i += 1
    
    return totalreward

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


def main():
  env = gym.make('MountainCar-v0')
  ft = FeatureTransformer(env)
  model = Model(env, ft, "constant")
  gamma = 0.99
  N = 300
  totalrewards = np.empty(N)
  for n in range(N):
    #reduce epsilon geometrically
    eps = 0.1*(0.97**n)
    totalreward = play_one(model, env, eps, gamma)
    totalrewards[n] = totalreward
    if (n + 1) % 100 == 0:
      print("episode:", n, "total reward:", totalreward)
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", -totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()
  plot_running_avg(totalrewards)

main()