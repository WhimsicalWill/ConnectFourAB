import numpy as np
import time
import torch
from utils import GameState, ConnectFourGame, ConnectFourGym
from evaluations import eval1, eval2
from networks import ValueNetwork, device

def minimax_move(true_state, max_depth=2, eval=None):
    """
    Return the minimax move (currently assumes it's player 1's turn)

    Args:
        max_depth - (int) the search depth
        eval - (function) the evaluation function to use at max-depth, non-terminal nodes

    Returns:
        best_score - (float) the best score attainable under minimax
        move - (int) the column that is optimal under the minimax search
    """
    def dfs(state, d, alpha, beta):
        status = state.get_game_over_status()
        if status is not None:
            return status, None
        if d == max_depth:
            if eval == None: # fixed evaluation function
                return eval2(state.b), None
            else: # learned evaluation function
                if state.turn == 1: # Max's turn
                    obs = torch.tensor(state.b, dtype=torch.float32).to(device)
                    return eval(obs).item(), None
                else: # Min's turn
                    obs = -torch.tensor(state.b, dtype=torch.float32).to(device)
                    return -eval(obs).item(), None

        moves = state.get_valid_moves()
        next_turn = state.get_next_turn()
        best_move = None
        if state.turn == 1: # Max's turn
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

    best_score, move = dfs(true_state, 0, float("-inf"), float("inf"))
    return best_score, move

def run_game(p1_depth, p2_depth, v_funcs=(None, None), sleep_time=1, debug=False, human=False):
    env = ConnectFourGym(6, 7)
    p1_value_func, p2_value_func = v_funcs
    state, done = env.reset(random_start=not human), False
    while not done:
        if state.turn == 1:
            best_score, action = minimax_move(state, max_depth=p1_depth, eval=p1_value_func)
        elif human: # When human = True, human controls player 2
            best_score, action = None, int(input("Enter a column (1-7) to drop your piece\n")) - 1
        else:
            best_score, action = minimax_move(state, max_depth=p2_depth, eval=p2_value_func)
        next_state, reward, done, _ = env.step(action)
        if debug:
            print(f"\nPlayer {state.turn} moved")
            print(f"Minimax score for move: {best_score}")
            print(f"Move number {env.game.get_num_moves()}, Player {next_state.turn} goes next\n")
            env.game.print_board()
            time.sleep(sleep_time)
        state = next_state
    print(f"Game ended. Final result: {reward}")
    return reward
    
def run_experiment(p1_depth, p2_depth, v_funcs=(None, None), num_episodes=100):
    p1_wins, p2_wins = 0, 0
    for _ in range(num_episodes):
        result = run_game(p1_depth, p2_depth, v_funcs)
        if result == -1:
            p2_wins += 1
        elif result == 1:
            p1_wins += 1
    ties = num_episodes - (p1_wins + p2_wins)
    print(f"p1_wins: {p1_wins}%, p2_wins: {p2_wins}%, ties: {ties}%")


if __name__ == "__main__":
    v_net = ValueNetwork((6, 7))
    v_net.load_checkpoint("data/model_1_07.pth")
    # run_game(2, 0, v_net, sleep_time=0, debug=True, human=True)
    # run_game(7, 7, sleep_time=0, debug=True)
    run_experiment(2, 2, v_funcs=(v_net, None))