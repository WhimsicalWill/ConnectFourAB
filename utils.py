import random
import gym
import copy
from gym import spaces

class InvalidMove(Exception):
    def __init__(self, message):
        self.message = message

class GameState:
    """
    Contains information about a game state
    """
    def __init__(self, board, turn, next=None, score=None):
        """
        Args:
            rows, cols - (int) the dimensions of the board
            b - (list) the board state
            turn - (int) 0 or 1, indicating the player to move
            next - (GameState) a pointer to the next state in the game
        """
        self.rows, self.cols = len(board), len(board[0])
        self.b = board
        self.turn = turn
        self.next = next

    def get_valid_moves(self):
        moves = []
        for c in range(self.cols):
            if self.b[0][c] == 0:
                moves.append(c)
        return moves

    def get_updated_board(self, col):
        """
        Assumes a valid move is given as input

        Returns:
            updated_board - (list) new board after move is made
        """
        updated_board = copy.deepcopy(self.b) # TODO: might not need deep
        for i in reversed(range(self.rows)):
            if updated_board[i][col] == 0:
                updated_board[i][col] = 1 if self.turn == 1 else -1
                break
        return updated_board

    def get_game_over_status(self):
        """
        Check for a win by checking for four consecutive cells with the same value in a row, column, or diagonal

        Returns:
            (int) - -1, 0, 1, or None. These represent a p1 loss, tie, p1 win, and ongoing game, respectively
        """
        def check_h(r, c):
            """Check for a horizontal win"""
            if c <= self.cols - 4:
                if self.b[r][c] != 0 and self.b[r][c] == self.b[r][c+1] == self.b[r][c+2] == self.b[r][c+3]:
                    return True

        def check_v(r, c):
            """Check for a vertical win"""
            if r <= self.rows - 4:
                if self.b[r][c] != 0 and self.b[r][c] == self.b[r+1][c] == self.b[r+2][c] == self.b[r+3][c]:
                    return True

        def check_md(r, c):
            """Check for a main diagonal win (top-left to bottom-right)"""
            if r <= self.rows - 4 and c <= self.cols - 4:
                if self.b[r][c] != 0 and self.b[r][c] == self.b[r+1][c+1] == self.b[r+2][c+2] == self.b[r+3][c+3]:
                    return True

        def check_od(r, c):
            """Check for an off diagonal win (bottom-left to top-right)"""
            if r >= 3 and c <= len(self.b[0]) - 4:
                if self.b[r][c] != 0 and self.b[r][c] == self.b[r-1][c+1] == self.b[r-2][c+2] == self.b[r-3][c+3]:
                    return True

        for r in range(self.rows):
            for c in range(self.cols):
                is_win = check_h(r, c) or check_v(r, c) or check_md(r, c) or check_od(r, c)
                if is_win:
                    return 1 if self.b[r][c] == 1 else -1
        
        # If no one won, then check for tie by checking if board is full
        for r in range(self.rows):
            for c in range(self.cols):
                if self.b[r][c] == 0:
                    return None
        return 0

    def get_next_turn(self):
        if self.turn == 1:
            return 2
        else:
            return 1

    def print_board(self):
        for row in self.b:
            for value in row:
                if value == -1:
                    value = 2
                print(value, end=' ')
            print()

class ConnectFourGame:
    """
    Defines fields and functions for a game of Connect Four
    """
    def __init__(self, rows, cols, p1_first=True):
        initial_turn = random.choice([1, 2])
        self.rows = rows
        self.cols = cols
        self.state = GameState(self.get_initial_board(), initial_turn)
        self.num_moves = 0

    def get_initial_board(self):
        return [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def make_move(self, col):
        """
        Drop piece in a given column and change the game state
        """
        if col < 0 or col >= self.cols:
            raise InvalidMove("Invalid column")
        if self.state.b[0][col] != 0:
            raise InvalidMove("Column is full")
        updated_board = self.state.get_updated_board(col)
        updated_turn = self.state.get_next_turn()
        prev_state = self.state 
        self.state = GameState(updated_board, updated_turn)
        prev_state.next = self.state
        self.num_moves += 1

    def get_turn(self):
        return self.state.turn

    def get_board(self):
        return self.state.board

    def status(self):
        return self.state.get_game_over_status()

    def get_random_move(self):
        return random.choice(self.state.get_valid_moves())

    def get_valid_moves(self):
        return self.state.get_valid_moves()
    
    def reset(self):
        initial_turn = random.choice([1, 2])
        self.state = GameState(self.get_initial_board(), initial_turn)
        self.num_moves = 0

    def print_board(self):
        self.state.print_board()

class ConnectFourGym(gym.Env):
    def __init__(self, rows, cols):
        self.game = ConnectFourGame(rows, cols)
        self.action_space = spaces.Discrete(cols)
        self.observation_space = spaces.Box(low=0, high=2, shape=(1, rows, cols), dtype=int)
        self.reward_range = (-1, 1)

        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):
        self.game.reset()
        return self.game.state

    def step(self, action):
        self.game.make_move(action)
        obs = self.game.state
        reward = self.game.status()
        done = reward != None
        reward = 0 if reward == None else reward
        info = {}
        return obs, reward, done, info
