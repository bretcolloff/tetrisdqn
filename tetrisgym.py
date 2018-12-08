import numpy as np
import random
from enum import Enum

class Direction(Enum):
    Left = 0
    Right = 1
    Down = 2

class Piece:
    """ Defines a single Tetris piece. """
    def __init__(self, type):
        self.type = type
        self.positions = self.init_position()

    def init_position(self):
        """ Sets the initial piece position after it's been generated. """

        # Starting from the top left, assuming 0, 0.
        if self.type is 'I':
            return [(1, 3), (1, 4), (1, 5), (1, 6)]
        elif self.type is 'J':
            return [(0, 3), (1, 3), (1, 4), (1, 5)]
        elif self.type is 'L':
            return [(0, 5), (1, 3), (1, 4), (1, 5)]
        elif self.type is 'O':
            return [(0, 4), (0, 5), (1, 4), (1, 5)]
        elif self.type is 'S':
            return [(0, 4), (0, 5), (1, 3), (1, 4)]
        elif self.type is 'T':
            return [(0, 4), (1, 3), (1, 4), (1, 5)]
        elif self.type is 'Z':
            return [(0, 3), (0, 4), (1, 4), (1, 5)]
        else:
            raise Exception("{} is not a real piece type.".format(self.type))

    def shifted(self, direction):
        """ Shift the piece in a direction, 0 is left, right is 1, down is 2 """
        new_positions = []
        if direction == Direction.Left:
            for y, x in self.positions:
                new_positions.append((y, x - 1))
        elif direction == Direction.Right:
            for y, x in self.positions:
                new_positions.append((y, x + 1))
        elif direction == Direction.Down:
            for y, x in self.positions:
                new_positions.append((y + 1, x))
        return new_positions


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

    def reset_game(self):
        """ Set up for a new iteration of the game. """
        self.board = np.zeros((self.height, self.width))
        self.score = 0

    def choose_piece(self):
        """ Decide which piece the player is going to get next. """
        piece_index = random.randint(0, len(self.pieces) - 1)
        return self.pieces[piece_index]

    def evalulate_piece_move(self, direction=Direction.Down):
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
            # Check to see if we've intersected with a block lower down.
            elif self.board[y][x] == 2:
                return 2
        return 0

    def update(self, action=None):
        self.remove_active_piece()

        if action is None:
            # Move the piece down one.
            # Check to see if it's gone down too far.
            # Do we need to generate a new piece?
            # Do we need to delete any rows?
            pass

        move = Direction.Down
        result = self.evalulate_piece_move(move)
        if result == 0:
            new_pos = self.piece.shifted(move)
            self.piece.positions = new_pos
            self.draw_piece()
        if result == 2:
            self.draw_piece(2)
            self.piece = Piece(self.choose_piece())


        self.steps = self.steps + 1

    def remove_active_piece(self):
        for i in range(len(self.board)):
            row = self.board[i]
            for j in range(len(row)):
                if row[j] == 1:
                    row[j] = 0

    def draw_piece(self, blocktype=1):
        """ We'll want to 'undraw' the piece to move it. We also might want to make it permanent.
            Set the piece to 0 to blank it, 1 to draw it active, 2 to draw it permanent. """
        for block in self.piece.positions:
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
                    piece = "_"
                elif block == 1:
                    piece = "O"
                elif block == 2:
                    piece = "#"
                else:
                    raise Exception("{} is an invalid piece type.".format(block))
                line = line + piece
            print(line)
        print ("Step {} - Score {}".format(self.steps, self.score))



