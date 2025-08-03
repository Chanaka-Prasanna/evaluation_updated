import os
import torch
import torch.nn as nn
import pickle

# -- Hyperparameters for GRU V3 Model --
MAX_LEN = 500        # max sequence length
EMBED_SIZE = 128
HIDDEN_DIM = 512     
NUM_LAYERS = 6       # increased from 4 (V2) to 6 (V3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- GRU Encoder-Decoder Model Definition --
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru       = nn.GRU(embed_size, hidden_dim, num_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        _, hidden = self.gru(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru       = nn.GRU(embed_size, hidden_dim, num_layers, batch_first=True)
        self.fc_out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, hidden):
        embedded   = self.embedding(tgt)
        outputs, h = self.gru(embedded, hidden)
        return self.fc_out(outputs), h

class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        hidden = self.encoder(src)
        logits, _ = self.decoder(tgt, hidden)
        return logits

# -- Tokenizer & Model Loading Utilities --
def load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, 'rb') as f:
        tok = pickle.load(f)
    return tok['stoi'], tok['itos']

def load_regex_to_e_nfa_model(model_path: str, tokenizer_path: str):
    stoi, itos = load_tokenizer(tokenizer_path)
    vocab_size = len(stoi)
    
    # Create encoder, decoder, and full model
    encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_DIM, NUM_LAYERS)
    decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_DIM, NUM_LAYERS)
    model = Seq2SeqGRU(encoder, decoder).to(device)
    
    # Load the full model state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, stoi, itos

# -- Encoding & Prediction --
def encode_sentence(s: str, stoi: dict, max_len: int = MAX_LEN):
    """GRU V3 encoding function matching the original notebook structure"""
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
    
    # Encode like the original: [SOS] + chars + [EOS] + padding
    seq = [sos] + [stoi.get(c, pad) for c in s][:max_len-2] + [eos]
    pad_tokens = [pad] * (max_len - len(seq))
    return torch.tensor(seq + pad_tokens, dtype=torch.long).unsqueeze(0)

def predict_regex_to_e_nfa(s: str, model: Seq2SeqGRU, stoi: dict, itos: dict,
            max_len: int = MAX_LEN, device: torch.device = device) -> str:
    """GRU V3 prediction function matching the original notebook"""
    model.eval()
    
    # Encode the input sequence
    src = encode_sentence(s, stoi, max_len).to(device)
    hidden = model.encoder(src)
    
    # Get SOS token with robust handling
    if '< SOS >' in stoi:
        sos_token = stoi['< SOS >']
    elif 'SOS' in stoi:
        sos_token = stoi['SOS']
    else:
        sos_token = 1  # fallback
    
    out_idxs = [sos_token]
    
    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_tensor = torch.tensor(out_idxs, device=device).unsqueeze(0)
            logits, hidden = model.decoder(tgt_tensor, hidden)
            nxt = logits[0, -1].argmax().item()
            
            # Check for EOS token (matching original code)
            if (itos[nxt] == '<EOS>' or 
                itos[nxt] == 'EOS' or 
                nxt == stoi.get('<EOS>', stoi.get('EOS', 2))):
                break
                
            out_idxs.append(nxt)
    
    # Return the generated sequence (excluding the initial SOS token)
    return ''.join(itos[i] for i in out_idxs[1:]) 