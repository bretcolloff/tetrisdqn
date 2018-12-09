import numpy as np
import random
from enum import Enum

class Move(Enum):
    Down = 0
    Left = 1
    Right = 2
    Rotate = 3

    def __int__(self):
        return self.value

piece_positions_map = {}
piece_positions_map["I"] = [
    [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
]
piece_positions_map["J"] = [
    [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
    [[1, 1, 0], [1, 0, 0], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1], [0, 0, 0]],
    [[0, 1, 0], [0, 1, 0], [1, 1, 0]],
]
piece_positions_map["L"] = [
    [[0, 0, 1], [1, 1, 1], [0, 0, 0]],
    [[1, 0, 0], [1, 0, 0], [1, 1, 0]],
    [[1, 1, 1], [1, 0, 0], [0, 0, 0]],
    [[1, 1, 0], [0, 1, 0], [0, 1, 0]],
]
piece_positions_map["O"] = [
    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]] # Pos 1
]
piece_positions_map["S"] = [
    [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
    [[1, 0, 0], [1, 1, 0], [0, 1, 0]]
]
piece_positions_map["T"] = [
    [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
    [[1, 0, 0], [1, 1, 0], [1, 0, 0]],
    [[1, 1, 1], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [1, 1, 0], [0, 1, 0]],
]
piece_positions_map["Z"] = [
    [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
    [[0, 1, 0], [1, 1, 0], [1, 0, 0]],
]

# Start positions
piece_corner_map = {}
piece_corner_map["I"] = (0, 3)
piece_corner_map["J"] = (0, 3)
piece_corner_map["L"] = (0, 3)
piece_corner_map["O"] = (0, 3)
piece_corner_map["S"] = (0, 3)
piece_corner_map["T"] = (0, 3)
piece_corner_map["Z"] = (0, 3)


class Piece:
    """ Defines a single Tetris piece. """
    def __init__(self, type):
        self.type = type
        self.init_position()

    def init_position(self):
        """ Sets the initial piece position after it's been generated. """
        self.states = piece_positions_map[self.type]
        self.state = 0
        self.state_corner_x, self.state_corner_y = (0, 0) # We need to set a starting point for these.
        return

    def shifted(self, direction):
        """ Shift the piece in a direction, 0 is left, right is 1, down is 2 """
        new_positions = []
        if direction == Move.Left:
            for y, x in self.get_position():
                new_positions.append((y, x - 1))
        elif direction == Move.Right:
            for y, x in self.get_position():
                new_positions.append((y, x + 1))
        elif direction == Move.Down:
            for y, x in self.get_position():
                new_positions.append((y + 1, x))
        elif direction == Move.Rotate:
            starting_state = self.state
            self.state = self.get_next_state()
            for y, x in self.get_position():
                new_positions.append((y, x))
            self.state = starting_state

        return new_positions

    def shift(self, direction):
        if direction == Move.Left:
            self.state_corner_x -= 1
        elif direction == Move.Right:
            self.state_corner_x += 1
        elif direction == Move.Down:
            self.state_corner_y += 1
        elif direction == Move.Rotate:
            self.state = self.get_next_state()

    def get_next_state(self):
        new_state = self.state + 1
        if new_state < len(self.states):
            return new_state
        else:
            return 0

    def get_position(self):
        self.state_shape = self.states[self.state]
        pos = []

        for i in range(len(self.state_shape)):
            row = self.state_shape[i]
            for j in range(len(row)):
                block = row[j]
                if block == 1:
                    pos.append((self.state_corner_y + i, self.state_corner_x + j))
        return pos


class TetrisGym:
    """ Defines the gym environment for the agent to learn to play Tetris in. """
    def __init__(self):
        # Standard tetris pieces.
        self.pieces = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']

        # Standard tetris board size.
        self.width = 10
        self.height = 20
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.board = None
        self.reset_game()
        self.piece = Piece(self.choose_piece())
        self.journey_buffer = []

    def reset_game(self):
        """ Set up for a new iteration of the game. """
        self.board = np.zeros((self.height, self.width))
        self.score = 0
        return self.board

    def choose_piece(self):
        """ Decide which piece the player is going to get next. """
        piece_index = random.randint(0, len(self.pieces) - 1)
        return self.pieces[piece_index]

    def evalulate_piece_move(self, direction=Move.Down):
        """ Evaluate the piece direction.
            0 = Left
            1 = Right
            2 = Down

            Return 0 if it's continuing to fall.
            Return 1 if it's an invalid lateral move.
            Return 2 if it's at the bottom of the screen. """

        new_position = self.piece.shifted(direction)


        for y, x in new_position:
            # Check to see if we've gone too far left.
            if x < 0:
                return 1
            # Check to see if we've gone too far right.
            elif x > self.width - 1:
                return 1
            # Check to see if we've gone too far down.
            elif y >= self.height:
                return 2
            # Check to see if we've intersected with another block.
            elif self.board[y][x] == 2 and y < 4:
                return 3
            elif self.board[y][x] == 2:
                return 2
        return 0

    def update(self, action=Move.Down):
        #next_state, reward, done, _ = gym.update(action)
        self.remove_active_piece()
        pre_action_state = np.copy(self.board)

        if action is not Move.Down:
            result = self.evalulate_piece_move(action)
            if result == 0:
                self.piece.shift(action)
            # Move the piece down one.
            # Check to see if it's gone down too far.
            # Do we need to generate a new piece?
            # Do we need to delete any rows?
            pass

        result = self.evalulate_piece_move(Move.Down)
        reward = 0
        piece_landed = False
        if result == 0:
            self.piece.shift(Move.Down)
            self.draw_piece()
        if result == 2:
            self.draw_piece(2)
            reward = self.evaluate_board(Move.Down)
            piece_landed = True
            self.piece = Piece(self.choose_piece())
        if result == 3:
            self.game_over = True

        next_state = np.copy(self.board)
        self.journey_buffer.append((pre_action_state, action, reward, next_state))

        self.steps = self.steps + 1
        if piece_landed:

            # We want to look at all the moves between the start of this piece and it landing, and evaluate it's path.
            # pre_action_state, action, reward, next_state
            experiences = self.walk_journey()
            self.journey_buffer = []
            return experiences, next_state, self.game_over
        else:
            return None, next_state, self.game_over

    def walk_journey(self):
        num_steps = len(self.journey_buffer)
        _, _, reward, _ = self.journey_buffer[-1]
        reward_step = reward / num_steps
        output_experiences = []

        for i in range(num_steps):
            experience = self.journey_buffer[i]
            state, action, exp_reward, next_state = experience
            experience = (state, action, reward_step * i+1, next_state)
            output_experiences.append(experience)
        return output_experiences

    def evaluate_board(self, action):
        completed_lines = []
        for i in range(len(self.board)):
            row = self.board[i]

            # Check to see if complete.
            complete = True
            for j in row:
                if j == 0:
                    complete = False
                    break

            if complete:
                completed_lines.append(i)

        num_completed = len(completed_lines)
        if num_completed > 4:
            num_completed = 4

        # Delete them bottom up or it's going to wreck the board.
        for line in reversed(completed_lines):
            self.board = np.delete(self.board, (line), axis=0)

        # Add all the lines back.
        empty_rows = np.zeros((num_completed, self.width))
        self.board = np.concatenate((empty_rows, self.board), axis=0)

        rows_score = 0
        self.score += rows_score

        occupied_rows = 0
        occupied_blocks = 0
        for rows in self.board:
            occupied_row = False
            for block in rows:
                if block == 2:
                    occupied_blocks += 1
                    occupied_row = True
            if occupied_row:
                occupied_rows += 1

        total_blocks_in_stack = occupied_rows * self.width
        ratio = occupied_blocks / total_blocks_in_stack
        ratio = ratio - 0.5
        return ratio

    def remove_active_piece(self):
        for i in range(len(self.board)):
            row = self.board[i]
            for j in range(len(row)):
                if row[j] == 1:
                    row[j] = 0

    def draw_piece(self, blocktype=1):
        """ We'll want to 'undraw' the piece to move it. We also might want to make it permanent.
            Set the piece to 0 to blank it, 1 to draw it active, 2 to draw it permanent. """
        for block in self.piece.get_position():
            y, x = block
            if self.board[y][x] == 2:
                print ("!")
            self.board[y][x] = blocktype

    def render(self):
        for row in self.board:
            line = ""
            for block in row:
                piece = ""
                if block == 0:
                    piece = "[_]"
                elif block == 1:
                    piece = "[X]"
                elif block == 2:
                    piece = "[#]"
                else:
                    raise Exception("{} is an invalid piece type.".format(block))
                line = line + piece
            print(line)
        print ("Step {} - Score {} - Shape - {}".format(self.steps, self.score, self.piece.type))
        if self.game_over:
            print ("GAME OVER")
