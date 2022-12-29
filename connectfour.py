import time
import random
from utils import GameState, ConnectFourGame
from evaluations import eval1, eval2

def minimax_move(game, max_depth=2, eval=eval2):
    """
    Return the minimax move (currently assumes it's player 0's turn)

    Args:
        b - (list) the 2D board state
        max_depth - (int) the search depth
        eval - (function) the evaluation function to use at max-depth, non-terminal nodes

    Returns:
        best_score - (float) the best score attainable under minimax
        move - (int) the column that is optimal under the minimax search
    """
    def dfs(state, d, alpha, beta):
        """
        Args:
            state - (GameState) game state
            d - (int) current depth
        """
        status = state.get_game_over_status()
        if status is not None:
            return status, None
        if d == max_depth:
            return eval(state.b), None

        moves = state.get_valid_moves()
        next_turn = (state.turn + 1) % 2
        best_move = None
        if state.turn == 0: # Max's turn
            best_score = float("-inf")
            for move in moves:
                updated_board = state.get_updated_board(move)
                next_state = GameState(updated_board, next_turn)
                score, _ = dfs(next_state, d + 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        else: # Min's turn
            best_score = float("inf")
            for move in moves:
                updated_board = state.get_updated_board(move)
                next_state = GameState(updated_board, next_turn)
                score, _ = dfs(next_state, d + 1, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
        return best_score, best_move

    best_score, move = dfs(game.state, 0, float("-inf"), float("inf"))
    return best_score, move

def run_game(p0_depth, p1_depth, sleep_time=1, debug=False, human=False):
    # p0_first = random.choice([True, False])
    game = ConnectFourGame(6, 7, p0_first)
    while game.status() == None:
        current_turn = game.get_turn()
        if current_turn == 0:
            best_score, move = minimax_move(game, max_depth=p0_depth)
        elif human: # When human = True, human controls player 1
            best_score, move = None, int(input("Enter a column (1-7) to drop your piece\n")) - 1
        else:
            best_score, move = minimax_move(game, max_depth=p1_depth)
        game.make_move(move)
        if debug:
            print(f"\nPlayer {current_turn} moved")
            print(f"Minimax score for move: {best_score}")
            print(f"Move number {game.num_moves}, Player {game.get_turn()} goes next\n")
            game.print_board()
            time.sleep(sleep_time)

    print(f"Game ended. Final result: {game.status()}")
    return game.status()
    
def run_experiment(p0_depth, p1_depth, num_episodes=100):
    p0_wins, p1_wins, ties = 0, 0, 0
    for ep in range(num_episodes):
        result = run_game(p0_depth, p1_depth)
        if result == -1:
            p1_wins += 1
        elif result == 1:
            p0_wins += 1
        else:
            ties += 1
    p0_wins = p0_wins * 100 / num_episodes
    p1_wins = p1_wins * 100 / num_episodes
    ties = ties * 100 / num_episodes
    print(f"Results (in %) for {num_episodes} episodes:")
    print(f"p0_wins: {p0_wins}%, p1_wins: {p1_wins}%, ties: {ties}%")


if __name__ == "__main__":
    # run_game(7, 7, sleep_time=0, debug=True)
    run_game(8, 8, sleep_time=0, debug=True, human=True)
    # run_experiment(6, 3)

# TODO: make some sort of dummy evaluation function (done)
# Implement minimax tree search (done)
# Implement alpha-beta pruning (done LOL)
# Think about MCTS implementation
