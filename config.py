class ConnectFourConfig:
    def __init__(
        self,
        train_steps,
        save_every,
        load_weights,
        target_update,
        search_depth,
        discount_factor,
        tau,
        save_path
    ):
        self.train_steps = train_steps
        self.save_every = save_every
        self.load_weights = load_weights
        self.target_update = target_update
        self.search_depth = search_depth
        self.discount_factor = discount_factor
        self.tau = tau
        self.save_path = save_path
