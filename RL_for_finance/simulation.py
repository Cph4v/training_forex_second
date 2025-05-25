#
# Monte Carlo Simulation Environment
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
#

import math
import random
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
# from generative_AI import SyntheticGenerator

rng = default_rng()


class ActionSpace:
    def sample(self):
        return random.randint(0, 1)


class Simulation:
    def __init__(self, symbol, feature, n_features,
                 start, end, periods,
                 min_accuracy=0.525, x0=100,
                 kappa=1, theta=100, sigma=0.2,
                 normalize=True, new=False):
        self.symbol = symbol
        self.feature = feature
        self.n_features = n_features
        self.start = start
        self.end = end
        self.periods = periods
        self.x0 = x0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.min_accuracy = min_accuracy
        self.normalize = normalize
        self.new = new
        self.action_space = ActionSpace()
        self._simulate_data()
        self._prepare_data()

    # def _simulate_data(self):
    #     index = pd.date_range(start=self.start,
    #                 end=self.end, periods=self.periods)
    #     s = [self.x0]
    #     dt = (index[-1] - index[0]).days / 365 / self.periods
    #     for t in range(1, len(index)):
    #         s_ = (s[t - 1] + self.kappa * (self.theta - s[t - 1]) * dt +
    #               s[t - 1] * self.sigma * math.sqrt(dt) * random.gauss(0, 1))
    #         s.append(s_)
        
    #     self.data = pd.DataFrame(s, columns=[self.symbol], index=index)
    #     print(f"====> : {self.data}")
    #     print(f"type : {type(self.data)}")
    

    def _simulate_data(self):
        n = 1
        raw = pd.read_parquet('/home/amir/RL/RL_platform/ML-Algotrading-Project-main (5)/ML-Algotrading-Project-main/dataset/data/stage_one_data/XAUUSD_stage_one.parquet').dropna()
        rets = raw['close'].iloc[-2 * 252:]
        rets = np.log((rets / rets.shift(n)).dropna())
        rets = rets.values
        # scaler = StandardScaler()
        # rets_ = scaler.fit_transform(rets.reshape(-1, 1))
        data = pd.DataFrame({'real': rets})
        noise = np.random.normal(0, 1, (len(rets), 1))
        generator = load_model('/home/amir/RL/RL_platform/models/generator.h5')
        synthetic_data = generator.predict(noise, verbose=False)
        # data[f'synth'] = scaler.inverse_transform(
        #                                         synthetic_data)

        data[f'synth'] = synthetic_data

        index = pd.date_range(start=self.start,
                    end=self.end, periods=self.periods - n)
        self.data = pd.DataFrame(data['synth'].values , columns=[self.symbol], index=index)
        print(f"=====> {self.data}")

    def _prepare_data(self):
        self.data['r'] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace=True)
        if self.normalize:
            self.mu = self.data.mean()
            self.std = self.data.std()
            self.data_ = (self.data - self.mu) / self.std
        else:
            self.data_ = self.data.copy()
        self.data['d'] = np.where(self.data['r'] > 0, 1, 0)
        self.data['d'] = self.data['d'].astype(int)

    def _get_state(self):
        return self.data_[self.feature].iloc[self.bar -
                                self.n_features:self.bar]
        
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_random_seed(seed)
        
    def reset(self):
        if self.new:
            self._simulate_data()
            self._prepare_data()
        self.treward = 0
        self.accuracy = 0
        self.bar = self.n_features
        state = self._get_state()
        return state.values, {}

    def step(self, action):
        if action == self.data['d'].iloc[self.bar]:
            correct = True
        else:
            correct = False
        reward = 1 if correct else 0 
        self.treward += reward
        self.bar += 1
        self.accuracy = self.treward / (self.bar - self.n_features)
        if self.bar >= len(self.data):
            done = True
        elif reward == 1:
            done = False
        elif (self.accuracy < self.min_accuracy and
              self.bar > self.n_features + 15):
            done = True
        else:
            done = False
        next_state = self.data_[self.feature].iloc[
            self.bar - self.n_features:self.bar].values
        return next_state, reward, done, False, {}

