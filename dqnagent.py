import numpy as np
import random
from tetrisgym import Move, TetrisGym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import os
import random


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

class DQN:
    def __init__(self):
        self.memory = deque(maxlen=1000)
        self.action_size = 4
        self.state_size = 200
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(-1, 200))
        return np.argmax(act_values[0])

    def integer_to_action(self, input):
        if input == 0:
            return Move.Down
        elif input == 1:
            return Move.Left
        elif input == 2:
            return Move.Right
        elif input == 3:
            return Move.Rotate

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            next_state = next_state.reshape(-1, 200)
            state = state.reshape(-1, 200)
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][int(action)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


gym = TetrisGym()
batch_size = 64
finished = False
agent = DQN()
scores = []

for episode in range(2000):
    gym = TetrisGym()
    current_state = gym.reset_game()
    runtime = 0
    while not gym.game_over:
        action = agent.act(current_state)
        #next_state, reward, done = gym.update(agent.integer_to_action(action))
        update, current_state, done = gym.update(agent.integer_to_action(action))
        #cls()
        #gym.render()

        if update is None:
            pass
        else:
            for state, action, reward, next_state in update:
                reward = reward
                next_state = next_state.reshape(-1)
                state = state.reshape(-1)
                agent.remember(state, action, reward, next_state, False)
                state = next_state
        if done:
            gym.render()
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(episode, 2000, runtime, agent.epsilon))
            scores.append(runtime)
            break
        else:
            runtime += 1
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

plt.plot(scores)
plt.ylabel('Steps Survived')
plt.xlabel('Episode')
plt.show()
print ("done!")
