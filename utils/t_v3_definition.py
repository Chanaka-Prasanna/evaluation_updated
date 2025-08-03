import os
import torch
import torch.nn as nn
import pickle

# -- Hyperparameters for V3 Model --
MAX_LEN = 500        # increased from 250 to 500
EMBED_SIZE = 128     
NUM_HEADS = 8        
NUM_ENCODER_LAYERS = 3  # back to 3 (from V2's 5)
NUM_DECODER_LAYERS = 3  # back to 3 (from V2's 5)
HIDDEN_DIM = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Positional Encoding --
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*(-torch.log(torch.tensor(10000.0))/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.pe = pe.unsqueeze(0).to(device)
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

# -- Seq2Seq Transformer Model Definition (V3) --
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.src_tok_emb = nn.Embedding(vocab_size, EMBED_SIZE)
        self.tgt_tok_emb = nn.Embedding(vocab_size, EMBED_SIZE)
        self.pos_enc = PosEnc(EMBED_SIZE)
        self.transformer = nn.Transformer(
            d_model=EMBED_SIZE,
            nhead=NUM_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=HIDDEN_DIM,
        )
        self.fc_out = nn.Linear(EMBED_SIZE, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.pos_enc(self.src_tok_emb(src))
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt))
        # square subsequent mask for teacher forcing
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(device)
        out = self.transformer(
            src_emb.permute(1, 0, 2),
            tgt_emb.permute(1, 0, 2),
            tgt_mask=tgt_mask
        )
        return self.fc_out(out.permute(1, 0, 2))

# -- Tokenizer & Model Loading Utilities --
def load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, 'rb') as f:
        tok = pickle.load(f)
    return tok['stoi'], tok['itos']

def load_regex_to_e_nfa_model(model_path: str, tokenizer_path: str):
    stoi, itos = load_tokenizer(tokenizer_path)
    vocab_size = len(stoi)
    model = Seq2SeqTransformer(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, stoi, itos

# -- Encoding & Prediction --
def encode_sentence(s: str, stoi: dict, max_len: int = MAX_LEN):
    """V3 encoding function with robust token handling"""
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
    
    # Tokenize the input sentence
    seq = [sos] + [stoi.get(c, pad) for c in s][:max_len-2] + [eos]
    seq += [pad] * (max_len - len(seq))
    return torch.tensor(seq).unsqueeze(0)

def predict_regex_to_e_nfa(s: str, model: Seq2SeqTransformer, stoi: dict, itos: dict,
            max_len: int = MAX_LEN, device: torch.device = device) -> str:
    """V3 prediction function adapted from the notebook"""
    model.eval()
    src = encode_sentence(s, stoi, max_len).to(device)
    
    # Get SOS token with same robust handling
    if '< SOS >' in stoi:
        sos_token = stoi['< SOS >']
    elif 'SOS' in stoi:
        sos_token = stoi['SOS']
    else:
        sos_token = 1  # fallback
    
    tgt = torch.tensor([[sos_token]], device=device)
    out_str = ''
    
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(src, tgt)
            next_tok = logits[0, -1].argmax().item()
            
            # Check for EOS token with robust handling
            if (itos[next_tok] == '<EOS>' or 
                itos[next_tok] == 'EOS' or 
                next_tok == stoi.get('<EOS>', stoi.get('EOS', 2))):
                break
                
            out_str += itos[next_tok]
            tgt = torch.cat([tgt, torch.tensor([[next_tok]], device=device)], dim=1)
    
    return out_str 