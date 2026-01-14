import streamlit as st
import pandas as pd
import os

from utils.t_v4_definition import load_regex_to_e_nfa_model, predict_regex_to_e_nfa
from evaluate import exact_match_rate, average_edit_distance, bleu, rouge_l, edit_distance


# Set page config
st.set_page_config(
    page_title="Regex to E-NFA Model Evaluation",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Regex to E-NFA Model Evaluation")
st.markdown("Upload your dataset and evaluate the model performance using various metrics.")

# Sidebar for model information
st.sidebar.header("Model Information")
st.sidebar.info("""
This app uses a Transformer-based model to convert regex patterns to E-NFA representations.

**Model: Transformer V4**
- 3 encoder layers, 3 decoder layers
- 8 attention heads
- Embedding size: 128
- Hidden dimension: 512
- Max sequence length: 250
- Character-level tokenization

**Metrics calculated:**
- Exact Match Rate
- Average Edit Distance
- BLEU Score
- ROUGE-L Score
""")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with columns for input and expected output"
    )
    
    if uploaded_file is not None:
        # Load and display dataset
        df = pd.read_csv(uploaded_file)
        st.success(f"Dataset loaded successfully! ({len(df)} rows)")
        
        # Show column selection
        st.subheader("Column Mapping")
        columns = df.columns.tolist()
        
        input_col = st.selectbox(
            "Select input column (regex patterns):",
            columns,
            index=0 if columns else None
        )
        
        output_col = st.selectbox(
            "Select target column (E-NFA representations):",
            columns,
            index=1 if len(columns) > 1 else 0
        )
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(), use_container_width=True)

with col2:
    st.header("🎯 Model Configuration")
    
    # Model and tokenizer paths
    model_path = st.text_input(
        "Model path:",
        value="model/transformer_regex_to_e_nfa.pt",
        help="Path to the trained model file"
    )
    
    tokenizer_path = st.text_input(
        "Tokenizer path:",
        value="model/regex_to_e_nfa_tokenizer.pkl",
        help="Path to the tokenizer file"
    )
    
    # Check if files exist
    model_exists = os.path.exists(model_path)
    tokenizer_exists = os.path.exists(tokenizer_path)
    
    st.write("**File Status:**")
    st.write(f"✅ Model: {model_path}" if model_exists else f"❌ Model: {model_path}")
    st.write(f"✅ Tokenizer: {tokenizer_path}" if tokenizer_exists else f"❌ Tokenizer: {tokenizer_path}")

# Evaluation section
if uploaded_file is not None and model_exists and tokenizer_exists:
    st.header("🚀 Run Evaluation")
    
    if st.button("Start Evaluation", type="primary", use_container_width=True):
        try:
            # Load model and tokenizer
            with st.spinner("Loading model and tokenizer..."):
                model, stoi, itos = load_regex_to_e_nfa_model(model_path, tokenizer_path)
                st.success("Model and tokenizer loaded successfully!")
            
            # Get data
            inputs = df[input_col].tolist()
            references = df[output_col].tolist()
            
            # Make predictions
            predictions = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, input_text in enumerate(inputs):
                status_text.text(f"Processing {i+1}/{len(inputs)}: {input_text[:50]}...")
                prediction = predict_regex_to_e_nfa(
                    str(input_text), model, stoi, itos
                )
                predictions.append(prediction)
                progress_bar.progress((i + 1) / len(inputs))
            
            status_text.text("Predictions completed!")
            
            # Calculate metrics
            st.header("📊 Evaluation Results")
            
            # Exact Match Rate
            exact_match = exact_match_rate(predictions, references)
            
            # Average Edit Distance
            avg_edit_dist = average_edit_distance(predictions, references)
            
            # BLEU Score (tokenize strings into characters)
            pred_tokens = [[c for c in pred] for pred in predictions]
            ref_tokens = [[c for c in ref] for ref in references]
            bleu_score = bleu(pred_tokens, ref_tokens)
            
            # ROUGE-L Score
            rouge_score = rouge_l(pred_tokens, ref_tokens)
            
            # Display metrics in columns
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    label="Exact Match Rate",
                    value=f"{exact_match:.3f}",
                    help="Fraction of predictions that exactly match the reference"
                )
            
            with metric_col2:
                st.metric(
                    label="Avg Edit Distance",
                    value=f"{avg_edit_dist:.2f}",
                    help="Average character-level edit distance"
                )
            
            with metric_col3:
                st.metric(
                    label="BLEU Score",
                    value=f"{bleu_score:.3f}",
                    help="BLEU score measuring n-gram overlap"
                )
            
            with metric_col4:
                st.metric(
                    label="ROUGE-L Score",
                    value=f"{rouge_score:.3f}",
                    help="ROUGE-L F-score based on longest common subsequence"
                )
            
            # Detailed results table
            st.header("📋 Detailed Results")
            
            results_df = pd.DataFrame({
                'Input': inputs,
                'Reference': references,
                'Prediction': predictions,
                'Match': [pred == ref for pred, ref in zip(predictions, references)],
                'Edit Distance': [edit_distance(pred, ref) for pred, ref in zip(predictions, references)]
            })
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                show_matches_only = st.checkbox("Show exact matches only")
            with col2:
                show_mismatches_only = st.checkbox("Show mismatches only")
            
            # Filter results
            filtered_df = results_df.copy()
            if show_matches_only:
                filtered_df = filtered_df[filtered_df['Match'] == True]
            elif show_mismatches_only:
                filtered_df = filtered_df[filtered_df['Match'] == False]
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    "Match": st.column_config.CheckboxColumn("Exact Match"),
                    "Edit Distance": st.column_config.NumberColumn("Edit Distance", format="%d")
                }
            )
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred during evaluation: {str(e)}")
            st.exception(e)

else:
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to start evaluation.")
    elif not model_exists or not tokenizer_exists:
        st.error("❌ Model or tokenizer files are missing. Please check the file paths.")

 