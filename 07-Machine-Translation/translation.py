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

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional = True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):

        embedded = self.dropout(self.embedding(src))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
                
        packed_outputs, hidden = self.rnn(packed_embedded)        

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        #outputs: [src len, batch size, hid dim * num directions]

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        #hidden: [batch size, hid dim]
        
        return outputs, hidden

# Define Attention
class AdditiveAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        self.v = nn.Linear(hid_dim, 1, bias = False)
        self.W = nn.Linear(hid_dim,     hid_dim) #for decoder
        self.U = nn.Linear(hid_dim * 2, hid_dim) #for encoder outputs
                
    def forward(self, hidden, encoder_outputs, mask):

        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #hidden = [batch size, src len, hid dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch size, src len, hid dim * 2]
        
        energy = torch.tanh(self.W(hidden) + self.U(encoder_outputs))
        #energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)
        #attention = [batch size, src len]

        attention = attention.masked_fill(mask, -1e10)
        
        return F.softmax(attention, dim = 1)

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        self.fc = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
      
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_outputs, mask)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim = 2)
   
        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc(torch.cat((output, weighted, embedded), dim = 1))
        
        return prediction, hidden.squeeze(0), a.squeeze(1)

# Define Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src == self.src_pad_idx).permute(1, 0)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
                    
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        attentions = torch.zeros(trg_len, batch_size, src.shape[0]).to(self.device)

        encoder_outputs, hidden = self.encoder(src, src_len)

        input_ = trg[0,:]
        
        mask = self.create_mask(src)
                
        for t in range(1, trg_len):

            output, hidden, attention = self.decoder(input_, hidden, encoder_outputs, mask)

            outputs[t] = output

            attentions[t] = attention

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1) 

            input_ = trg[t] if teacher_force else top1
            
        return outputs, attentions

# Define the function to translate from Myanmar to English
def translate(source):
    src_text = text_transform[SRC_LANGUAGE](source).to(device)
    src_text = src_text.reshape(-1, 1)  #because batch_size is 1
    text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)
    
    target = "<pad>"*int(src_text[0]*2)
    trg_text = text_transform[TRG_LANGUAGE](target).to(device)
    trg_text = trg_text.reshape(-1, 1)

    input_dim   = len(vocab_transform[SRC_LANGUAGE])
    output_dim  = len(vocab_transform[TRG_LANGUAGE])
    emb_dim     = 64  
    hid_dim     = 128  
    dropout     = 0.5

    attn = AdditiveAttention(hid_dim)
    enc  = Encoder(input_dim,  emb_dim,  hid_dim, dropout)
    dec  = Decoder(output_dim, emb_dim,  hid_dim, dropout, attn)

    SRC_PAD_IDX = PAD_IDX

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

    model.load_state_dict(torch.load('./models/AdditiveSeq2Seq.pt'))

    model.eval()
    with torch.no_grad():
        output, attentions = model(src_text, text_length, trg_text, 0) #turn off teacher forcing

    output = output.squeeze(1)

    output = output[1:]

    output_max = output.argmax(1)

    mapping = vocab_transform[TRG_LANGUAGE].get_itos()

    translation = []
    for token in output_max:
        translation.append(mapping[token.item()])
    
    return print(' '.join(translation))