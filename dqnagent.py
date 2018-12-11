import numpy as np
from tetrisgym import Move, TetrisGym, TETRIS_HEIGHT, TETRIS_WIDTH
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import random
import time

class DQN:
    def __init__(self):
        self.memory = deque(maxlen=10000)
        self.action_size = 4
        self.state_size = 200
        self.improvement_rate = 0.95
        self.randomness = 1.0
        self.randomness_min = 0.01
        self.randomness_decay = 0.999
        self.learning_rate = 0.001
        self.model = self.compile_model()

    def compile_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                         activation='relu',
                         input_shape=(TETRIS_HEIGHT, TETRIS_WIDTH, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape((1, TETRIS_HEIGHT, TETRIS_WIDTH, 1)))
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
            next_state = next_state
            state = state
            old_q = self.model.predict(state.reshape((1, TETRIS_HEIGHT, TETRIS_WIDTH, 1)))
            new_q = self.model.predict(next_state.reshape((1, TETRIS_HEIGHT, TETRIS_WIDTH, 1)))

            if not done:
                scaled_old = (1 - self.improvement_rate) * old_q[0][int(action)]
                scaled_new = self.improvement_rate * (reward + new_q[0][int(action)])
                target = scaled_old + scaled_new

                old_q[0][int(action)] = target
            self.model.fit(state.reshape((1, TETRIS_HEIGHT, TETRIS_WIDTH, 1)), old_q, epochs=1, verbose=0)
        if self.randomness > self.randomness_min:
            self.randomness *= self.randomness_decay

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
        update, current_state, done = gym.update(agent.integer_to_action(action))
        gym.render()
        time.sleep(0.01)

        if update is None:
            pass
        else:
            for state, action, reward, next_state in update:
                reward = reward
                next_state = next_state
                state = state
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
