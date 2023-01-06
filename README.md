Implementation of minimax tree search and alpha-beta pruning with a learned value function. Implementation works for discrete zero-sum games, and is tested on ConnectFour.

The value function is learned, and is updated using an n-step reward objective. The method is on-policy, and only updates using experience from the current episode.