# Import necessary libraries
import torch, torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import pyidaungsu.tokenize as tokenize
import random, math, time
import pickle

# Choose the computing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set SEED for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Define the source and target languages
SRC_LANGUAGE = 'MY'
TRG_LANGUAGE = 'EN'

# Creat tokenizers
token_transform = {}

def myanmar_tokenizer(text):
    return tokenize(text, form='word')

token_transform[SRC_LANGUAGE] = myanmar_tokenizer
token_transform[TRG_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# Load vocab
with open('vocab_transform.pkl', 'rb') as file:
    vocab_transform = pickle.load(file)

# Define the function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# Define the function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

# Define the function to convert texts in src and trg languages from raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor



