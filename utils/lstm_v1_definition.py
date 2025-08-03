import os
import torch
import torch.nn as nn
import pickle

# -- Hyperparameters for LSTM V1 Model --
MAX_LEN = 500        # max sequence length (updated from 200 to 500)
EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- LSTM Encoder-Decoder Model Definition --
class Encoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED_SIZE)
        self.rnn = nn.LSTM(EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
    def forward(self, x):
        e = self.emb(x)
        _, (h, c) = self.rnn(e)
        return h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED_SIZE)
        self.rnn = nn.LSTM(EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, vocab_size)
    def forward(self, x, h, c):
        e = self.emb(x)                     # x shape: [batch, seq_len]
        out, (h, c) = self.rnn(e, (h, c))   # out: [batch, seq_len, HIDDEN_SIZE]
        return self.fc(out), h, c           # logits: [batch, seq_len, vocab_size]

# -- Tokenizer & Model Loading Utilities --
def load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, 'rb') as f:
        tok = pickle.load(f)
    return tok['stoi'], tok['itos']

def load_regex_to_e_nfa_model(model_path: str, tokenizer_path: str):
    stoi, itos = load_tokenizer(tokenizer_path)
    vocab_size = len(stoi)
    
    # Create encoder and decoder
    encoder = Encoder(vocab_size).to(device)
    decoder = Decoder(vocab_size).to(device)
    
    # Load the checkpoint (contains both encoder and decoder state dicts)
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['enc'])
    decoder.load_state_dict(checkpoint['dec'])
    
    encoder.eval()
    decoder.eval()
    
    # Return as a tuple to match interface expectations
    return (encoder, decoder), stoi, itos

# -- Encoding & Prediction --
def encode_sentence(s: str, stoi: dict, max_len: int = MAX_LEN):
    """LSTM encoding function matching the original notebook structure"""
    # Debug: Check available special tokens
    special_tokens = [k for k in stoi.keys() if k.startswith('<') or 'SOS' in k or 'EOS' in k or 'PAD' in k]
    print(f"Available special tokens: {special_tokens}")
    
    # Try different possible token names for PAD
    if '<PAD>' in stoi:
        pad = stoi['<PAD>']
    elif 'PAD' in stoi:
        pad = stoi['PAD']
    else:
        pad = 0  # fallback to 0
    
    # Try different possible token names for SOS
    if '< SOS >' in stoi:
        sos = stoi['< SOS >']
    elif 'SOS' in stoi:
        sos = stoi['SOS']
    else:
        sos = 1  # fallback to 1
    
    # Try different possible token names for EOS
    if '<EOS>' in stoi:
        eos = stoi['<EOS>']
    elif 'EOS' in stoi:
        eos = stoi['EOS']
    else:
        eos = 2  # fallback to 2
    
    # Encode exactly like the original: [SOS] + chars + [EOS] + padding
    seq = [sos] + [stoi.get(c, pad) for c in s][:max_len-2] + [eos]
    seq += [pad] * (max_len - len(seq))
    return torch.tensor(seq).unsqueeze(0)

def predict_regex_to_e_nfa(s: str, model_tuple, stoi: dict, itos: dict,
            max_len: int = MAX_LEN, device: torch.device = device) -> str:
    """LSTM prediction function matching the original notebook"""
    # Unpack encoder and decoder from the model tuple
    encoder, decoder = model_tuple
    
    encoder.eval()
    decoder.eval()
    
    # Encode the input sequence (matching the original encode function)
    src = encode_sentence(s, stoi, max_len).to(device)
    h, c = encoder(src)
    
    # Get SOS token with robust handling
    if '< SOS >' in stoi:
        sos_token = stoi['< SOS >']
    elif 'SOS' in stoi:
        sos_token = stoi['SOS']
    else:
        sos_token = 1  # fallback
    
    inputs = torch.tensor([[sos_token]], device=device)
    out_str = ''
    
    with torch.no_grad():
        for _ in range(max_len):
            logits, h, c = decoder(inputs, h, c)
            tok = logits.argmax(-1).item()
            
            # Check for EOS token (matching original code)
            if (itos[tok] == '<EOS>' or 
                itos[tok] == 'EOS' or 
                tok == stoi.get('<EOS>', stoi.get('EOS', 2))):
                break
                
            out_str += itos[tok]
            inputs = torch.tensor([[tok]], device=device)
    
    
    return out_str 