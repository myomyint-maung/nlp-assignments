# Import necessary libraries
import pytreebank
import torch, torchtext
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import FastText
from torch.utils.data   import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

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

# Create the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout):
        super().__init__()
        #put padding_idx so asking the embedding layer to ignore padding
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, 
                           hid_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, output_dim)
        
    def forward(self, text, text_lengths):
        #text = [batch size, seq len]
        embedded = self.embedding(text)
        
        #++ pack sequence ++
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False, batch_first=True)
        
        #embedded = [batch size, seq len, embed dim]
        packed_output, (hn, cn) = self.lstm(packed_embedded)  #if no h0, all zeroes
        
        #++ unpack in case we need to use it ++
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        #output = [batch size, seq len, hidden dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        #hn = [batch size, hidden dim * num directions]
        
        return self.fc(hn)

#explicitly initialize weights for better learning
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

# Set the parameters for the LSTM model
input_dim  = len(vocab)
hid_dim    = 256
emb_dim    = 300
output_dim = 2

#for biLSTM
num_layers = 2
bidirectional = True
dropout = 0.5

model = LSTM(input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout).to(device)
model.apply(initialize_weights)
model.embedding.weight.data = fast_embedding #**<------applied the fast text embedding as the initial weights

# Set the hyperparameters for training
lr=1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

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

# Create a function to calculate prediction accuracy
def accuracy(preds, y):
    
    predicted = torch.max(preds.data, 1)[1]
    batch_corr = (predicted == y).sum()
    acc = batch_corr / len(y)
    
    return acc

# Create a function to train the model
def train(model, loader, optimizer, criterion, loader_length):
    epoch_loss = 0
    epoch_acc = 0
    model.train() #useful for batchnorm and dropout
    
    for i, (label, text, text_length) in enumerate(loader): 
        label = label.to(device) #(batch_size, )
        text = text.to(device) #(batch_size, seq len)
                
        #predict
        predictions = model(text, text_length).squeeze(1) #output by the fc is (batch_size, 1), thus need to remove this 1
        
        #calculate loss
        loss = criterion(predictions, label)
        acc  = accuracy(predictions, label)
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
                        
    return epoch_loss / loader_length, epoch_acc / loader_length

# Create a function to evaluate the model
def evaluate(model, loader, criterion, loader_length):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for i, (label, text, text_length) in enumerate(loader): 
            label = label.to(device) #(batch_size, )
            text  = text.to(device)  #(seq len, batch_size)

            predictions = model(text, text_length).squeeze(1) 
            
            loss = criterion(predictions, label)
            acc  = accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / loader_length, epoch_acc / loader_length

# Train and save the model
best_val_loss = float('inf')
num_epochs      = 10

save_path = f'models/{model.__class__.__name__}.pt'

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):

    train_loss, train_acc = train(model, train_loader, optimizer, criterion, train_loader_length)
    val_loss, val_acc = evaluate(model, val_loader, criterion, val_loader_length)
    
    #for plotting
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)