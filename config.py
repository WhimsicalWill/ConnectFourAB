class ConnectFourConfig:
    def __init__(
        self,
        train_steps,
        log_interval,
        load_weights,
        target_update,
        search_depth,
        discount_factor,
        tau,
        save_path
    ):
        self.train_steps = train_steps
        self.log_interval = log_interval
        self.load_weights = load_weights
        self.target_update = target_update
        self.search_depth = search_depth
        self.discount_factor = discount_factor
        self.tau = tau
        self.save_path = save_path
