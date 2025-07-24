class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='loss', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        assert mode in ['loss', 'acc'], "mode must be either 'loss' or 'acc'"
        self.mode = mode

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return

        improve = (score < self.best_score - self.min_delta) if self.mode == 'loss' else (score > self.best_score + self.min_delta)

        if improve:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience} (best: {self.best_score:.4f})")
            if self.counter >= self.patience:
                print("🔴 Early stopping triggered")
                self.early_stop = True
