# Import libraries
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras

# Load saved recommendation models
embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
sentences = pickle.load(open('models/sentences.pkl', 'rb'))

# Ensure SentenceTransformer tokenizer has pad_token
rec_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Example model
if rec_model.tokenizer.pad_token is None:
    rec_model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    rec_model.resize_token_embeddings(len(rec_model.tokenizer))

# Load saved prediction models
loaded_model = keras.models.load_model("models/model.h5")

# Load the configuration of the text vectorizer
with open("models/text_vectorizer_config.pkl", "rb") as f:
    saved_text_vectorizer_config = pickle.load(f)

saved_text_vectorizer_config['output_mode'] = 'int'
if 'batch_input_shape' in saved_text_vectorizer_config:
    del saved_text_vectorizer_config['batch_input_shape']

loaded_text_vectorizer = TextVectorization.from_config(saved_text_vectorizer_config)

# Load the saved vocabulary
with open("models/vocab.pkl", "rb") as f:
    loaded_vocab = pickle.load(f)
loaded_text_vectorizer.set_vocabulary(loaded_vocab)

expected_input_dim = 158812
if len(loaded_vocab) != expected_input_dim:
    print(f"Warning: Vocabulary size {len(loaded_vocab)} does not match model input size {expected_input_dim}.")

def reshape_vectorized_output(vectorized_output, target_shape):
    current_shape = vectorized_output.shape[-1]
    if current_shape < target_shape:
        vectorized_output = np.pad(vectorized_output, ((0, 0), (0, target_shape - current_shape)))
    elif current_shape > target_shape:
        vectorized_output = vectorized_output[:, :target_shape]
    return vectorized_output

def recommendation(input_paper):
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper, convert_to_tensor=True))
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
    papers_list = [] 
    for i in top_similar_papers.indices:
        title=sentences[i.item()]
        abstract=sentences[i.item()][4]
        papers_list.append((title, abstract))

    return papers_list

def invert_multi_hot(encoded_labels):
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(loaded_vocab, hot_indices)

def predict_category(abstract, model, vectorizer, label_lookup):
    vectorized_text = vectorizer([abstract]).numpy()
    vectorized_text = reshape_vectorized_output(vectorized_text, expected_input_dim)
    predictions = model.predict(vectorized_text)
    predicted_labels = label_lookup(np.round(predictions).astype(int)[0])
    return predicted_labels

# Streamlit App
st.title('Research Papers Recommendation and Subject Area Prediction App')
st.write("LLM and Deep Learning Base App")

input_paper = st.text_input("Enter Paper title.....")
new_abstract = st.text_area("Paste paper abstract....")

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

if st.button("Recommend"):
    try:
        recommend_papers = recommendation(input_paper)
        st.subheader("Recommended Papers")
        st.write(recommend_papers)
        for title, abstract in recommend_papers:
            st.write(f"**Title**: {title}")
            st.write(f"**Abstract**: {abstract}")
            st.write("===================================================================")
    except Exception as e:
        st.error(f"Error in recommendation: {e}")

    st.write("===================================================================")

    try:
        predicted_categories = predict_category(new_abstract, loaded_model, loaded_text_vectorizer, invert_multi_hot)
        st.subheader("Predicted Subject Area")
        st.write(predicted_categories)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
    
    
    if not st.session_state.feedback_submitted:
        st.subheader("Your Feedback")
        feedback_rating = st.slider("Rate the Recommendations (1-5)", 1, 5, 3)
        feedback_comment = st.text_area("Provide your comments or suggestions")

        if st.button("Submit Feedback"):
            # Save feedback logic goes here, like saving to a database or file
            st.session_state.feedback_submitted = True  # Mark feedback as submitted
            st.write(f"Thank you for your feedback!")
            st.write(f"Rating: {feedback_rating} out of 5")
            st.write(f"Comment: {feedback_comment}")

    else:
        # Show feedback submission confirmation message
        st.write("Feedback Submitted! Thank you for your input.")
    
    st.write("===================================================================")