# Import necessary libraries
import pytreebank
import torch, torchtext
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import FastText
from torch.utils.data   import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Choose the computing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set SEED for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load the Stanford Sentiment Treebank dataset
sst = pytreebank.load_sst()

# Extract the training, validation and test sets
train  = sst['train']
val    = sst['dev']
test   = sst['test']

# Extract the training, validation and test data
datasets    = [train, val, test]
train_data  = []
val_data    = []
test_data   = []
data        = [train_data, val_data, test_data]

for i in range(len(data)):
  dataset = datasets[i]
  for example in dataset:
    for sentiment, text in example.to_labeled_lines():
      data[i].append((sentiment, text))

# Create a tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Create the vocab
def yield_tokens(data_iter):
    for data in data_iter:
        for _, text in data.to_labeled_lines():
            yield tokenizer(text)
        
vocab = build_vocab_from_iterator(yield_tokens(train), specials=['<unk>','<pad>','<bos>','<eos>'])

# Set <unk> as the default index of the vocab
vocab.set_default_index(vocab['<unk>'])

# Set the padding index
pad_idx = vocab['<pad>']

# Load FastText embeddings
fast_vectors = FastText(language='simple')
fast_embedding = fast_vectors.get_vecs_by_tokens(vocab.get_itos()).to(device)

# Create text and label pipelines
text_pipeline  = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

# Create a function to collate batches
def collate_batch(batch):
    label_list, text_list, length_list = [], [], []
    for (_label, _text) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        length_list.append(processed_text.size(0))
        
    return torch.tensor(label_list, dtype=torch.int64), \
        pad_sequence(text_list, padding_value=pad_idx, batch_first=True), \
        torch.tensor(length_list, dtype=torch.int64)

# Prepare dataloaders
batch_size = 64

train_loader = DataLoader(train_data, batch_size = batch_size,
                          shuffle=True, collate_fn=collate_batch)

val_loader   = DataLoader(val_data, batch_size = batch_size,
                          shuffle=True, collate_fn=collate_batch)

test_loader  = DataLoader(test_data, batch_size = batch_size,
                          shuffle=True, collate_fn=collate_batch)

# Define the lengths of the data loaders
train_loader_length = len(list(iter(train_loader)))
val_loader_length   = len(list(iter(val_loader)))
test_loader_length  = len(list(iter(test_loader)))

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

# The model's parameters
input_dim  = len(vocab)
hid_dim    = 256
emb_dim    = 300
output_dim = 5
num_layers = 2
bidirectional = True
dropout = 0.5

model = LSTM(input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout).to(device)

# Load the train model
save_path = f'models/LSTM.pt'
model.load_state_dict(torch.load(save_path))

# Prediction
def prediction(text_list):
    result = list()
    for text in text_list:
        text = torch.tensor(text_pipeline(text)).to(device)
        text = text.reshape(1, -1)
        text_length = torch.tensor([text.size(1)]).to(dtype=torch.int64)
        output = model(text, text_length).squeeze(1)
        predicted = torch.max(output.data, 1)[1].detach().cpu().numpy()[0]
        result.append((test_str, predicted))
    return result