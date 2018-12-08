from tetrisgym import Move, TetrisGym
import os
import time
import random


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

manual = True
gym = TetrisGym()

while gym.game_over is False:
    #cls()

    if not manual:
        # Do some random movement tests so we can check exents.
        dir = random.randint(0, 2)
        if dir == 0:
            gym.update()
        elif dir == 1:
            gym.update(action=Move.Left)
        elif dir == 2:
            gym.update(action=Move.Right)
    else:
        move = input("l, r, a, empty")
        if move == "l":
            gym.update(action=Move.Left)
        elif move == "r":
            gym.update(action=Move.Right)
        elif move == "a":
            gym.update(action=Move.Rotate)
        else:
            gym.update()
    gym.render()

    if not manual:
        time.sleep(1)
