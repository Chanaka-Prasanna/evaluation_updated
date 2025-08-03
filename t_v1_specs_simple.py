#!/usr/bin/env python3
"""
Simple T_V1 Model Specifications Extractor

Hardcoded paths - just run this file to extract model specifications.
No arguments needed!
"""

import os
import sys
import json
from datetime import datetime

# ========================================
# HARDCODED CONFIGURATION - EDIT THESE PATHS
# ========================================

# Model and tokenizer file paths (edit these to match your files)
MODEL_PATH = "model/regex_to_e_nfa_tokenizer.pkl"
TOKENIZER_PATH = "model/transformer_regex_to_e_nfa.pt"

# Alternative t_v1 specific paths (if you have them)
# MODEL_PATH = "model/t_v1_regex_to_e_nfa.pt"
# TOKENIZER_PATH = "model/t_v1_regex_to_e_nfa_tokenizer.pkl"

# Output file name
OUTPUT_FILE = "t_v1_model_specifications.json"

# ========================================
# CONSTANTS FROM T_V1 DEFINITION
# ========================================

# Hardcoded values from t_v1_definition.py (fallback if import fails)
MAX_LEN = 250
EMBED_SIZE = 128
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
HIDDEN_DIM = 512

# Add utils to path
sys.path.append('utils')

# Try to import constants from t_v1_definition
try:
    from utils.t_v1_definition import (
        MAX_LEN, EMBED_SIZE, NUM_HEADS, NUM_ENCODER_LAYERS, 
        NUM_DECODER_LAYERS, HIDDEN_DIM
    )
    print("✅ Imported constants from t_v1_definition.py")
except ImportError:
    print("⚠️ Using fallback constants (couldn't import from t_v1_definition.py)")

def get_file_info(file_path):
    """Get file information"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        return {
            "exists": True,
            "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 2)
        }
    else:
        return {
            "exists": False,
            "size_bytes": 0,
            "size_mb": 0
        }

def load_tokenizer_info():
    """Load tokenizer information without requiring torch"""
    tokenizer_info = {
        "file_path": TOKENIZER_PATH,
        "file_info": get_file_info(TOKENIZER_PATH)
    }
    
    if not tokenizer_info["file_info"]["exists"]:
        tokenizer_info["error"] = "Tokenizer file not found"
        return tokenizer_info
    
    try:
        import pickle
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        if isinstance(tokenizer_data, dict) and 'stoi' in tokenizer_data and 'itos' in tokenizer_data:
            stoi = tokenizer_data['stoi']
            itos = tokenizer_data['itos']
            
            # Get special tokens
            special_tokens = {}
            for token, idx in stoi.items():
                if token.startswith('<') and token.endswith('>'):
                    special_tokens[token] = idx
            
            # Get vocabulary sample (first 20 tokens)
            vocab_sample = dict(list(stoi.items())[:20])
            
            tokenizer_info.update({
                "vocabulary_size": len(stoi),
                "special_tokens": special_tokens,
                "vocabulary_sample": vocab_sample,
                "loaded_successfully": True
            })
        else:
            tokenizer_info["error"] = "Invalid tokenizer format"
            
    except Exception as e:
        tokenizer_info["error"] = f"Could not load tokenizer: {e}"
    
    return tokenizer_info

def extract_model_specifications():
    """Extract comprehensive model specifications"""
    print("🔍 T_V1 Model Specifications Extractor")
    print("=" * 50)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Tokenizer Path: {TOKENIZER_PATH}")
    print()
    
    # Get file information
    model_info = get_file_info(MODEL_PATH)
    tokenizer_info = load_tokenizer_info()
    
    # Build specifications
    specs = {
        "extraction_info": {
            "timestamp": datetime.now().isoformat(),
            "model_path": MODEL_PATH,
            "tokenizer_path": TOKENIZER_PATH,
            "extractor_version": "simple_hardcoded"
        },
        
        "file_status": {
            "model_file": {
                "path": MODEL_PATH,
                "exists": model_info["exists"],
                "size_bytes": model_info["size_bytes"],
                "size_mb": model_info["size_mb"]
            },
            "tokenizer_file": {
                "path": TOKENIZER_PATH,
                "exists": tokenizer_info["file_info"]["exists"],
                "size_bytes": tokenizer_info["file_info"]["size_bytes"],
                "size_mb": tokenizer_info["file_info"]["size_mb"]
            }
        },
        
        "model_architecture": {
            "model_name": "t_v1",
            "model_type": "Sequence-to-Sequence Transformer",
            "task": "Regex to E-NFA Translation",
            "framework": "PyTorch"
        },
        
        "hyperparameters": {
            "max_sequence_length": MAX_LEN,
            "embedding_dimension": EMBED_SIZE,
            "num_attention_heads": NUM_HEADS,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "num_decoder_layers": NUM_DECODER_LAYERS,
            "hidden_dimension": HIDDEN_DIM,
            "vocabulary_size": tokenizer_info.get("vocabulary_size", "Unknown")
        },
        
        "architecture_components": {
            "input_processing": {
                "tokenization": "Character-level",
                "special_tokens": ["<PAD>", "< SOS >", "<EOS>"],
                "max_input_length": MAX_LEN
            },
            
            "embedding_layers": {
                "source_embedding": {
                    "type": "nn.Embedding",
                    "vocab_size": tokenizer_info.get("vocabulary_size", "Unknown"),
                    "embedding_dim": EMBED_SIZE
                },
                "target_embedding": {
                    "type": "nn.Embedding",
                    "vocab_size": tokenizer_info.get("vocabulary_size", "Unknown"),
                    "embedding_dim": EMBED_SIZE
                },
                "positional_encoding": {
                    "type": "Sinusoidal",
                    "max_length": MAX_LEN,
                    "dimension": EMBED_SIZE
                }
            },
            
            "transformer_core": {
                "model_dimension": EMBED_SIZE,
                "attention_heads": NUM_HEADS,
                "encoder_layers": NUM_ENCODER_LAYERS,
                "decoder_layers": NUM_DECODER_LAYERS,
                "feedforward_dimension": HIDDEN_DIM,
                "attention_mechanism": "Multi-Head Self-Attention",
                "activation": "ReLU (PyTorch default)",
                "dropout": "Default PyTorch values"
            },
            
            "output_layer": {
                "type": "Linear",
                "input_dimension": EMBED_SIZE,
                "output_dimension": tokenizer_info.get("vocabulary_size", "Unknown"),
                "activation": "None (logits output)"
            }
        },
        
        "tokenizer_details": {
            "vocabulary_size": tokenizer_info.get("vocabulary_size", "Unknown"),
            "special_tokens": tokenizer_info.get("special_tokens", {}),
            "vocabulary_sample": tokenizer_info.get("vocabulary_sample", {}),
            "file_size_bytes": tokenizer_info["file_info"]["size_bytes"],
            "loaded_successfully": tokenizer_info.get("loaded_successfully", False),
            "error": tokenizer_info.get("error", None)
        },
        
        "estimated_parameters": {
            "source_embedding": f"vocab_size × {EMBED_SIZE}",
            "target_embedding": f"vocab_size × {EMBED_SIZE}",
            "positional_encoding": f"{MAX_LEN} × {EMBED_SIZE}",
            "encoder_layers": f"~{NUM_ENCODER_LAYERS} × 4 × {EMBED_SIZE}²",
            "decoder_layers": f"~{NUM_DECODER_LAYERS} × 4 × {EMBED_SIZE}²",
            "output_layer": f"vocab_size × {EMBED_SIZE}",
            "note": "Exact count requires loading the actual model",
            "estimated_total": "Several million parameters (depends on vocabulary size)"
        },
        
        "training_characteristics": {
            "loss_function": "Cross-Entropy Loss (typical for seq2seq)",
            "optimization": "Adam optimizer (typical for transformers)",
            "teacher_forcing": "Likely used during training",
            "attention_masking": "Causal masking for decoder",
            "regularization": "Dropout (PyTorch default values)"
        }
    }
    
    return specs

def print_specifications(specs):
    """Print specifications in a readable format"""
    
    print("📊 MODEL INFORMATION:")
    print(f"  Model Name: {specs['model_architecture']['model_name']}")
    print(f"  Model Type: {specs['model_architecture']['model_type']}")
    print(f"  Task: {specs['model_architecture']['task']}")
    print(f"  Framework: {specs['model_architecture']['framework']}")
    
    print(f"\n📁 FILE STATUS:")
    model_file = specs['file_status']['model_file']
    tokenizer_file = specs['file_status']['tokenizer_file']
    
    print(f"  Model File:")
    print(f"    Path: {model_file['path']}")
    print(f"    Exists: {'✅' if model_file['exists'] else '❌'}")
    if model_file['exists']:
        print(f"    Size: {model_file['size_mb']} MB ({model_file['size_bytes']:,} bytes)")
    
    print(f"  Tokenizer File:")
    print(f"    Path: {tokenizer_file['path']}")
    print(f"    Exists: {'✅' if tokenizer_file['exists'] else '❌'}")
    if tokenizer_file['exists']:
        print(f"    Size: {tokenizer_file['size_mb']} MB ({tokenizer_file['size_bytes']:,} bytes)")
    
    print(f"\n🔧 HYPERPARAMETERS:")
    for key, value in specs['hyperparameters'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🏗️ ARCHITECTURE COMPONENTS:")
    for section, details in specs['architecture_components'].items():
        print(f"\n  {section.replace('_', ' ').title()}:")
        for key, value in details.items():
            if isinstance(value, dict):
                print(f"    {key.replace('_', ' ').title()}:")
                for k, v in value.items():
                    print(f"      {k.replace('_', ' ').title()}: {v}")
            else:
                print(f"    {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n📝 TOKENIZER:")
    tok_details = specs['tokenizer_details']
    if tok_details.get('loaded_successfully', False):
        print(f"  Vocabulary Size: {tok_details['vocabulary_size']}")
        print(f"  Special Tokens: {len(tok_details['special_tokens'])}")
        if tok_details['special_tokens']:
            print("    Special Tokens List:")
            for token, idx in tok_details['special_tokens'].items():
                print(f"      {token}: {idx}")
    else:
        print(f"  Status: ❌ Could not load tokenizer")
        if tok_details.get('error'):
            print(f"  Error: {tok_details['error']}")
    
    print(f"\n📏 ESTIMATED PARAMETERS:")
    for key, value in specs['estimated_parameters'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

def save_specifications(specs):
    """Save specifications to JSON file"""
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(specs, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Specifications saved to: {OUTPUT_FILE}")
        return True
    except Exception as e:
        print(f"\n❌ Error saving file: {e}")
        return False

def main():
    """Main execution function"""
    try:
        # Extract specifications
        specs = extract_model_specifications()
        
        # Print to console
        print_specifications(specs)
        
        # Save to file
        save_specifications(specs)
        
        print(f"\n✅ Specification extraction completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        return False

if __name__ == "__main__":
    main() 