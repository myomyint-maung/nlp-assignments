# Import necessary libraries
import pytreebank
import torch, torchtext
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import FastText

# Choose the computing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the training data
sst = pytreebank.load_sst()
train  = sst['train']

train_data  = []
for example in train:
    for sentiment, text in example.to_labeled_lines():
            train_data.append((sentiment, text))

# Create the vocab
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens(data_iter):
    for data in data_iter:
        for _, text in data.to_labeled_lines():
            yield tokenizer(text)
        
vocab = build_vocab_from_iterator(yield_tokens(train), specials=['<unk>','<pad>','<bos>','<eos>'])

# Set <unk> as the default index of the vocab
vocab.set_default_index(vocab['<unk>'])

# Set the padding index
pad_idx = vocab['<pad>']

# Load FastText embeddings for the vocab
fast_vectors = FastText(language='simple')
fast_embedding = fast_vectors.get_vecs_by_tokens(vocab.get_itos()).to(device)

# The LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, 
                           hid_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, output_dim)
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False, batch_first=True)
        
        packed_output, (hn, cn) = self.lstm(packed_embedded)
        
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        
        return self.fc(hn)