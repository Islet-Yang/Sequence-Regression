import torch

class EarlyStopping:
    def __init__(self, patience=30, checkpoint_path='best_model.pth', mode='max'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        if mode == 'max':
            self.monitor_op = torch.gt
            self.best_score = float('-inf')
        elif mode == 'min':
            self.monitor_op = torch.lt
            self.best_score = float('inf')
        else:
            raise ValueError("Mode should be 'max' or 'min'.")

    def step(self, score, model):
        score = torch.tensor(score)
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

    def should_stop(self):
        return self.early_stop
      
    def now_count(self):
      return self.counter
      
