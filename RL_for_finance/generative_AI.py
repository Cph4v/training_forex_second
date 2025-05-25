import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

import os
import numpy as np
import pandas as pd
from pylab import plt, mpl


plt.style.use('seaborn-v0_8')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_generator(hu=32):
    model = Sequential()
    model.add(Dense(hu, activation='relu', input_dim=1))
    model.add(Dense(hu, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def create_discriminator(hu=32):
    model = Sequential()
    model.add(Dense(hu, activation='relu', input_dim=1))
    model.add(Dense(hu, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(),
                  metrics=['accuracy'])
    return model

def create_generator(hu=32):
    model = Sequential()
    model.add(Dense(hu, activation='relu', input_dim=1))
    model.add(Dense(hu, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def create_discriminator(hu=32):
    model = Sequential()
    model.add(Dense(hu, activation='relu', input_dim=1))
    model.add(Dense(hu, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(),
                  metrics=['accuracy'])
    return model

def create_gan(generator, discriminator, lr=0.001):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=lr))
    return model

generator = create_generator()
discriminator = create_discriminator()
gan = create_gan(generator, discriminator, 0.0001)

from numpy.random import default_rng

rng = default_rng(seed=100)

def train_models(y_, epochs, batch_size):
    for epoch in range(epochs):
        # Generate synthetic data
        noise = rng.normal(0, 1, (batch_size, 1))
        synthetic_data = generator.predict(noise, verbose=False)

        # Train discriminator
        real_data = y_[rng.integers(0, len(y_), batch_size)]
        discriminator.train_on_batch(real_data, np.ones(batch_size))
        discriminator.train_on_batch(synthetic_data,
                                     np.zeros(batch_size))

        # Train generator
        noise = rng.normal(0, 1, (batch_size, 1))
        gan.train_on_batch(noise, np.ones(batch_size))

        # Print progress
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}')
    return real_data, synthetic_data


raw = pd.read_parquet('/home/amir/RL/RL_platform/ML-Algotrading-Project-main (5)/ML-Algotrading-Project-main/dataset/data/stage_one_data/XAUUSD_stage_one.parquet').dropna()

rets = raw['close'].iloc[-2 * 252:]
rets = np.log((rets / rets.shift(1)).dropna())
rets = rets.values

scaler = StandardScaler()


rets_ = scaler.fit_transform(rets.reshape(-1, 1))
rng = default_rng(100)
tf.random.set_seed(100)


generator = create_generator(hu=24)
discriminator = create_discriminator(hu=24)
gan = create_gan(generator, discriminator, lr=0.0001)

generator.save('/home/amir/RL/RL_platform/models/generator.h5')
discriminator.save('/home/amir/RL/RL_platform/models/generator.h5')
gan.save('/home/amir/RL/RL_platform/models/generator.h5')

rd, sd = train_models(y_=rets_, epochs=5001, batch_size=32)
data = pd.DataFrame({'real': rets})

N = 25
from tensorflow.keras.models import load_model

def _simulate():
    raw = pd.read_parquet('/home/amir/RL/RL_platform/ML-Algotrading-Project-main (5)/ML-Algotrading-Project-main/dataset/data/stage_one_data/XAUUSD_stage_one.parquet').dropna()
    rets = raw['close'].iloc[-2 * 252:]
    rets = np.log((rets / rets.shift(1)).dropna())
    rets = rets.values
    # rets_ = scaler.fit_transform(rets.reshape(-1, 1))
    data = pd.DataFrame({'real': rets})
    scaler = StandardScaler()
    noise = np.random.normal(0, 1, (len(rets_), 1))
    generator = load_model('/home/amir/RL/RL_platform/RL_for_finance/generator_model.h5')
    synthetic_data = generator.predict(noise, verbose=False)
    data[f'synth'] = scaler.inverse_transform(
                                            synthetic_data)
    return data['synth'].values


for i in range(N):
    noise = np.random.normal(0, 1, (len(rets_), 1))
    synthetic_data = generator.predict(noise, verbose=False)
    data[f'synth_{i:02d}'] = scaler.inverse_transform(
                                            synthetic_data)

data.iloc[:, :2].plot(style=['r', 'b--', 'b--'], lw=1, alpha=0.7)
plt.show()

data['real'].plot(kind='hist', bins=50, label='real',
                  color='r', alpha=0.7)
data['synth_00'].plot(kind='hist', bins=50, alpha=0.7,
                  label='synthetic', color='b', sharex=True)
plt.legend()
plt.show()

plt.plot(np.sort(data['real']), 'r', lw=1.0, label='real')
plt.plot(np.sort(data['synth_00']), 'b--', lw=1.0, label='synthetic')
plt.legend()
plt.show()


sn = N
data.iloc[:, 1:sn + 1].cumsum().apply(np.exp).plot(
    style='b--', lw=0.7, legend=False)
data.iloc[:, 1:sn + 1].mean(axis=1).cumsum().apply(
    np.exp).plot(style='g', lw=2)
data['real'].cumsum().apply(np.exp).plot(style='r', lw=2)
plt.show()

# <img src="https://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>
# 
# <a href="https://tpq.io" target="_blank">https://tpq.io</a> | <a href="https://twitter.com/dyjh" target="_blank">@dyjh</a> | <a href="mailto:team@tpq.io">team@tpq.io</a>


#
# Synthetic Financial Data Generator using GAN
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
#

# import os
# import math
# import numpy as np
# import pandas as pd
# from numpy.random import default_rng
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import StandardScaler
# from scipy import stats
# from pylab import plt, mpl

# plt.style.use('seaborn-v0_8')
# mpl.rcParams['figure.dpi'] = 300
# mpl.rcParams['savefig.dpi'] = 300
# mpl.rcParams['font.family'] = 'serif'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# rng = default_rng(100)
# tf.random.set_seed(100)


# class SyntheticGenerator:
#     def __init__(self, data_url, column='close', seed=100, hu=24,
#                 epochs=1001, batch_size=32, n_samples=25, load=False, model_path='models'):
#         self.column = column
#         self.seed = seed
#         self.hu = hu
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.n_samples = n_samples
#         self.data_url = data_url
#         self._load_data()
#         self._prepare_data()
#         self._build_models()
#         if load:
#             self.load_models(model_path)
#         else:
#             self._train_models()
#         self._generate_synthetic()
#         self._train_models()
#         self._generate_synthetic()

#     # def _load_data(self):
#     #     raw = pd.read_csv(self.data_url, index_col=0, parse_dates=True).dropna()
#     #     self.rets = raw[self.column].iloc[-2 * 252:]
#     #     self.rets = np.log((self.rets / self.rets.shift(1)).dropna()).values

#     def _load_data(self):
#         raw = pd.read_parquet('/home/amir/RL/RL_platform/ML-Algotrading-Project-main (5)/ML-Algotrading-Project-main/dataset/data/stage_one_data/XAUUSD_stage_one.parquet').dropna()
#         print(f"dataset : {raw}")
#         self.rets = raw[self.column].iloc[-2 * 252:]
#         self.rets = np.log((self.rets / self.rets.shift(1)).dropna()).values

#     def _prepare_data(self):
#         self.scaler = StandardScaler()
#         self.rets_ = self.scaler.fit_transform(self.rets.reshape(-1, 1))

#     def _build_generator(self):
#         model = Sequential([
#             Dense(self.hu, activation='relu', input_shape=(1,)),
#             Dense(self.hu, activation='relu'),
#             Dense(1)
#         ])
#         return model

#     def _build_discriminator(self):
#         model = Sequential([
#             Dense(self.hu, activation='relu', input_shape=(1,)),
#             Dense(self.hu, activation='relu'),
#             Dense(1, activation='sigmoid')
#         ])
#         model.compile(optimizer=Adam(), loss='binary_crossentropy')
#         return model

#     def _build_gan(self, generator, discriminator, lr=0.0001):
#         discriminator.trainable = False
#         gan = Sequential([generator, discriminator])
#         gan.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')
#         return gan

#     def _build_models(self):
#         self.generator = self._build_generator()
#         self.discriminator = self._build_discriminator()
#         self.gan = self._build_gan(self.generator, self.discriminator)

#     def _train_models(self):
#         y_ = self.rets_
#         for epoch in range(self.epochs):
#             idx = rng.integers(0, len(y_), self.batch_size)
#             real = y_[idx]
#             noise = rng.normal(0, 1, (self.batch_size, 1))
#             fake = self.generator.predict(noise, verbose=False)

#             x = np.vstack((real, fake))
#             y = np.array([1] * self.batch_size + [0] * self.batch_size)
#             self.discriminator.trainable = True
#             self.discriminator.train_on_batch(x, y)

#             noise = rng.normal(0, 1, (self.batch_size, 1))
#             y2 = np.ones((self.batch_size, 1))
#             self.discriminator.trainable = False
#             self.gan.train_on_batch(noise, y2)

#     def _generate_synthetic(self):
#         self.data = pd.DataFrame({'real': self.rets.flatten()})
#         for i in range(self.n_samples):
#             noise = rng.normal(0, 1, (len(self.rets_), 1))
#             synth = self.generator.predict(noise, verbose=False)
#             self.data[f'synth_{i:02d}'] = self.scaler.inverse_transform(synth)

#     def save_models(self, path='models'):
#         os.makedirs(path, exist_ok=True)
#         self.generator.save('/home/amir/RL/RL_platform/RL_for_finance/generator.h5')
#         self.discriminator.save('/home/amir/RL/RL_platform/RL_for_finance/discriminator.h5')
#         self.gan.save('/home/amir/RL/RL_platform/RL_for_finance/gan.h5')

#     def load_models(self, path='models'):
#         from tensorflow.keras.models import load_model
#         self.generator = load_model('/home/amir/RL/RL_platform/RL_for_finance/generator.h5')
#         self.discriminator = load_model('/home/amir/RL/RL_platform/RL_for_finance/discriminator.h5')
#         self.gan = load_model('/home/amir/RL/RL_platform/RL_for_finance/gan.h5')


#     def evaluate(self):
#         pvals = []
#         for i in range(self.n_samples):
#             pval = stats.ks_2samp(self.data['real'], self.data[f'synth_{i:02d}']).pvalue
#             pvals.append(pval)
#         self.pvals = np.array(pvals)
#         return np.mean(self.pvals > 0.05)

#     def plot_distributions(self):
#         self.data['real'].plot(kind='hist', bins=50, alpha=0.7, color='r', label='real')
#         self.data['synth_00'].plot(kind='hist', bins=50, alpha=0.7, color='b', label='synthetic')
#         plt.legend()
#         plt.show()

#     def plot_sorted(self):
#         plt.plot(np.sort(self.data['real']), 'r', lw=1.0, label='real')
#         plt.plot(np.sort(self.data['synth_00']), 'b--', lw=1.0, label='synthetic')
#         plt.legend()
#         plt.show()

#     def plot_paths(self):
#         sn = self.n_samples
#         self.data.iloc[:, 1:sn + 1].cumsum().apply(np.exp).plot(style='b--', lw=0.7, legend=False)
#         self.data.iloc[:, 1:sn + 1].mean(axis=1).cumsum().apply(np.exp).plot(style='g', lw=2)
#         self.data['real'].cumsum().apply(np.exp).plot(style='r', lw=2)
#         plt.show()

#     def plot_pvals(self):
#         plt.hist(self.pvals, bins=100)
#         plt.axvline(0.05, color='r')
#         plt.show()


# # Example Usage

# if __name__ == '__main__':
#     url = 'https://certificate.tpq.io/rl4finance.csv'
#     synth = SyntheticGenerator(data_url=url)
#     synth.save_models()
#     ks_score = synth.evaluate()
#     print(f'Proportion of synthetic samples passing KS test: {ks_score:.2%}')
#     synth.plot_distributions()
#     synth.plot_sorted()
#     synth.plot_paths()
#     synth.plot_pvals()
