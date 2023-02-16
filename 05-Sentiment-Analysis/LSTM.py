# Import necessary libraries
import torch, torchtext
import torch.nn as nn
import pytreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import FastText

# Choose the computing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

