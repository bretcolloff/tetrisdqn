from tetrisgym import Direction, TetrisGym
import os
import time
import random


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


gym = TetrisGym()

while gym.game_over is False:
    cls()

    # Do some random movement tests so we can check exents.
    dir = random.randint(0, 2)
    if dir == 0:
        gym.update()
    elif dir == 1:
        gym.update(action=Direction.Left)
    elif dir == 2:
        gym.update(action=Direction.Right)
    gym.render()
    time.sleep(1)
