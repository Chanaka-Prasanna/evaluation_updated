# Regex to E-NFA Model Evaluation App

A Streamlit web application for evaluating the performance of a Transformer-based model that converts regex patterns to E-NFA representations.

## Features

- **File Upload**: Upload CSV datasets with regex patterns and expected E-NFA outputs
- **Model Prediction**: Generate predictions using the trained transformer model
- **Comprehensive Evaluation**: Calculate multiple metrics:
  - Exact Match Rate
  - Average Edit Distance (Levenshtein)
  - BLEU Score
  - ROUGE-L Score
- **Interactive Results**: View detailed results with filtering options
- **Export Results**: Download evaluation results as CSV

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Ensure you have the model files in the correct location:
   - `model/transformer_regex_to_e_nfa.pt` (trained model)
   - `model/regex_to_e_nfa_tokenizer.pkl` (tokenizer)

## Usage

1. Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

2. Open your web browser and go to the displayed URL (usually `http://localhost:8501`)

3. Upload your CSV dataset:

   - The CSV should contain at least two columns
   - One column for input regex patterns
   - One column for expected E-NFA representations

4. Configure the model:

   - Verify the model and tokenizer file paths
   - Select the correct input and output columns from your dataset

5. Run the evaluation:
   - Click "Start Evaluation" to begin processing
   - View real-time progress and results
   - Download the detailed results if needed

## Dataset Format

Your CSV file should have the following structure:

```csv
input_regex,expected_enfa
a+,{expected E-NFA representation}
(a|b)*,{expected E-NFA representation}
...
```

## Metrics Explained

- **Exact Match Rate**: Fraction (0-1) of predictions that exactly match the reference
- **Average Edit Distance**: Average number of character-level edits needed to transform prediction to reference
- **BLEU Score**: Measures n-gram overlap between predictions and references
- **ROUGE-L Score**: F-score based on longest common subsequence

## Troubleshooting

- Ensure model and tokenizer files exist in the specified paths
- Check that your CSV has the correct format and column names
- Make sure all required packages are installed

## Files Structure

```
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── utils/
│   └── model_definition.py   # Model classes and utilities
├── evaluate.py               # Evaluation metrics
└── model/
    ├── transformer_regex_to_e_nfa.pt      # Trained model
    └── regex_to_e_nfa_tokenizer.pkl       # Tokenizer
```
