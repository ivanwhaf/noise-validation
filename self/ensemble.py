"""
Ensemble model
Ensemble prediction
"""


class EnsembleModel(object):
    """
    Ensemble Model (Mean Teacher)
    """

    def __init__(self, model, beta):
        self.model = model
        self.beta = beta
        self.ensemble_model = {}  # mean teacher
        self.backup = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ensemble_model[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                average = self.beta * self.ensemble_model[name] + (1.0 - self.beta) * param.data
                self.ensemble_model[name] = average.clone()

    def apply_ensemble(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.ensemble_model[name]

    def restore_model(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class EnsemblePrediction(object):
    """
    Ensemble Predictions
    pred: numpy array len_dataset*num_classes
    """

    def __init__(self, pred, beta):
        self.pred = pred
        self.beta = beta
        self.ensemble_pred = {}
        self.backup = {}

        self.ensemble_pred = self.pred

    def update(self):
        self.ensemble_pred = self.beta * self.ensemble_pred + (1 - self.beta) * self.pred

    def apply_ensemble(self):
        self.backup = self.pred
        self.pred = self.ensemble_pred

    def restore_pred(self):
        self.pred = self.backup
        self.backup = {}
