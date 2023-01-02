Implementation of minimax tree search and alpha-beta pruning with a learned value function. Implementation works for discrete zero-sum games, and is tested on ConnectFour.

The value function is learned, and is updated using an n-step reward objective.

# TODO
# 1) write training loop (done)
# 2) use target value network for updates
# 3) train regression objective against eval2 function
# 4) track training progress with wandb
# 5) figure out how to benchmark results