from tetrisgym import Move, TetrisGym
import random
import time

manual = True
gym = TetrisGym()

while gym.game_over is False:
    if not manual:
        # Do some random movement tests so we can check exents.
        dir = random.randint(0, 2)
        if dir == 0:
            gym.update()
        elif dir == 1:
            gym.update(action=Move.Left)
        elif dir == 2:
            gym.update(action=Move.Right)
        time.sleep(0.1)
    else:
        move = input("l, r, a, empty")
        if move == "l": # Left
            gym.update(action=Move.Left)
        elif move == "r": # Right
            gym.update(action=Move.Right)
        elif move == "a": # Rotate
            gym.update(action=Move.Rotate)
        else:
            gym.update()
    gym.render()
