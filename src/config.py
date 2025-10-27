import torch
from enum import Enum

class Mode(Enum):
    NONE = 0
    TRAIN = 1
    TEST = 2
    DEMO = 3


# ---- Configuration ----

MODE = Mode.TRAIN # Modes for running the script: TRAIN, TEST, DEMO (and NONE)
DEBUGGING = False # True to display extra information about intermediate steps
PRETRAINED_MODEL = "./cosqa-finetuned-model-api"

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'