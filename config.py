class ConnectFourConfig:
    def __init__(
        self,
        train_steps,
        save_every,
        load_weights,
        target_update,
        search_depth,
        save_path
    ):
        self.train_steps = train_steps
        self.save_every = save_every
        self.load_weights = load_weights
        self.target_update = target_update
        self.search_depth = search_depth
        self.save_path = save_path
