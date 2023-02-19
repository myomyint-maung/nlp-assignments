# Import necessary libraries
import torch, torchtext, torchdata
from torch import nn

# Choose computing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set SEED for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

