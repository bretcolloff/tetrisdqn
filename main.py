from tetrisgym import TetrisGym
import os
import time


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


gym = TetrisGym()

while gym.game_over is False:
    cls()
    gym.update()
    gym.render()
    time.sleep(1)
