import pickle
import torch
import torchtext
import torch.nn as nn

# Choose the computing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the tokenizer
tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')

# Load the vocab
with open('vocab.pkl', 'rb') as file:
    vocab = pickle.load(file)

# Define the model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):     
        super().__init__()
        self.hid_dim   = hid_dim
        self.num_layers= num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm      = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers,
                                 dropout=dropout_rate, batch_first=True)
        self.dropout   = nn.Dropout(dropout_rate)
        self.fc        = nn.Linear(hid_dim, vocab_size)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell   = cell.detach()
        return hidden, cell
        
    def forward(self, src, hidden):
        embed = self.embedding(src)
        output, hidden = self.lstm(embed, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)       
        return prediction, hidden

# Instantiate the model 
vocab_size = len(vocab)
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65              
lr = 1e-3                     

model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)

# Load the pretrained model
save_path = f'models/lstm_lm.pt'
model.load_state_dict(torch.load(save_path, map_location=torch.device(device)))

# Define the function for code generation
def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    
    return tokens

# Define the function for prediction
def predict(prompt):
    max_seq_len = 30
    temperature = 0.5
    seed = 0
    generation = generate(prompt, max_seq_len, temperature, model, tokenizer, 
                          vocab, device, seed)
    return ' '.join(generation)

# Test the function for prediction
prompt = 'from sklearn.preprocessing '
print(predict(prompt))