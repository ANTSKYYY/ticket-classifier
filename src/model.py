import torch
import torch.nn as nn
from transformers import Trainer

class WeightedTrainer(Trainer):
    """Trainer avec CrossEntropyLoss pondérée pour gérer le déséquilibre de classes."""
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss