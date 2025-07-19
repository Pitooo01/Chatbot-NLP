# ------------------------------- MODEL 1 -------------------------------
# from flask import Flask, request, jsonify, render_template
# import pickle
# import json
# import faiss
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import os

# app = Flask(__name__)

# # Load model & metadata
# with open('model/intent_model.pkl', 'rb') as f:
#     clf_pipeline = pickle.load(f)

# with open('model/meta.json') as f:
#     meta = json.load(f)

# embed_model = SentenceTransformer(meta['model_name'])

# # Constants
# CRISIS_RESPONSE = meta['topic_default_ans']['suicide_selfharm']
# CRISIS_KEYWORDS = meta['crisis_keywords']
# SIM_THRESHOLD = meta['sim_threshold']
# TOPIC_DEFAULT_ANS = meta['topic_default_ans']
# FALLBACK_ANSWER = "Maaf, aku belum yakin. Bisa jelaskan lebih detail atau topiknya apa?"

# # Load corpus & FAISS
# df = pd.read_csv("model/mentalhealth_clean.csv")
# corpus_questions = df['Questions'].tolist()
# corpus_answers   = df['Answers'].tolist()
# corpus_topics    = df['topic'].tolist()
# corpus_urgent    = df['urgent'].tolist()

# index = faiss.read_index("model/corpus.index")

# def crisis_check(text: str):
#     return any(kw in text.lower() for kw in CRISIS_KEYWORDS)

# def retrieve(query: str, top_k=3):
#     q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
#     D, I = index.search(q_emb, top_k)
#     results = []
#     for sc, ix in zip(D[0], I[0]):
#         results.append({
#             'score': float(sc),
#             'answer': corpus_answers[ix],
#             'topic': corpus_topics[ix],
#             'urgent': bool(corpus_urgent[ix])
#         })
#     return results

# def chatbot_answer(user_text: str, top_k=3):
#     if crisis_check(user_text):
#         return CRISIS_RESPONSE, 'crisis'

#     hits = retrieve(user_text, top_k=top_k)
#     if hits and hits[0]['score'] >= SIM_THRESHOLD:
#         return hits[0]['answer'], hits[0]['topic']

#     try:
#         topic_pred = clf_pipeline.predict([user_text])[0]
#         return TOPIC_DEFAULT_ANS.get(topic_pred, FALLBACK_ANSWER), topic_pred
#     except:
#         return FALLBACK_ANSWER, 'unknown'

# # ========== Routes ==========
# @app.route('/')
# def home():
#     if os.path.exists("templates/index_true.html"):
#         return render_template('index_true.html')
#     return "index_true.html not found in /templates", 404

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get("message", "")
#     if not user_input.strip():
#         return jsonify({"reply": "Pertanyaannya kosong!"})
#     reply, topic = chatbot_answer(user_input)
#     return jsonify({"reply": reply, "topic": topic})

# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({"status": "ok"})

# if __name__ == "__main__":
#     app.run(debug=True, port=8000)

# ------------------------------- MODEL 2 ------------------------------- 

# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import re
# import os
# import json
# from keras.models import load_model, Model
# from keras.layers import Input

# app = Flask(__name__)

# # ========== Load Model ==========
# training_model = load_model('training_model.h5')

# # Cek apakah file ada dan tidak kosong
# if not os.path.exists('input_features_dict.json') or os.path.getsize('input_features_dict.json') == 0:
#     raise ValueError("File input_features_dict.json tidak ditemukan atau kosong!")

# # Load JSON
# with open('input_features_dict.json') as f:
#     input_features_dict = json.load(f)

# # with open('input_features_dict.json') as f:
# #     input_features_dict = json.load(f)

# with open('target_features_dict.json') as f:
#     target_features_dict = json.load(f)

# with open('reverse_target_features_dict.json') as f:
#     reverse_target_features_dict = json.load(f)

# num_encoder_tokens = len(input_features_dict)
# num_decoder_tokens = len(target_features_dict)
# max_encoder_seq_length = 20  # sesuaikan dengan modelmu
# max_decoder_seq_length = 20
# latent_dim = 256

# # ========== Build Encoder Model ==========
# encoder_inputs = training_model.input[0]
# _, state_h_enc, state_c_enc = training_model.layers[2].output
# encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

# # ========== Build Decoder Model ==========
# decoder_inputs = training_model.input[1]
# decoder_lstm = training_model.layers[3]
# decoder_dense = training_model.layers[4]

# # Tambahkan nama khusus untuk Input baru
# decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
# decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_input_c')
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# # Gunakan decoder_inputs dari model training sebelumnya
# decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
# decoder_outputs = decoder_dense(decoder_outputs)

# # Buat decoder_model dengan input yang sekarang namanya unik
# decoder_model = Model([decoder_inputs] + decoder_states_inputs,
#                       [decoder_outputs] + [state_h, state_c])


# # ========== Helper Function ==========
# def preprocess_input(text):
#     tokens = re.findall(r"[\w']+|[^\s\w]", text)
#     input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
#     for t, token in enumerate(tokens):
#         if token in input_features_dict:
#             input_matrix[0, t, input_features_dict[token]] = 1.
#     return input_matrix

# def decode_response(test_input):
#     states_value = encoder_model.predict(test_input)
#     target_seq = np.zeros((1, 1, num_decoder_tokens))
#     target_seq[0, 0, target_features_dict['<START>']] = 1.

#     decoded_sentence = ''
#     while True:
#         output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_token = reverse_target_features_dict[str(sampled_token_index)]

#         if sampled_token == '<END>' or len(decoded_sentence.split()) > max_decoder_seq_length:
#             break

#         decoded_sentence += ' ' + sampled_token
#         target_seq = np.zeros((1, 1, num_decoder_tokens))
#         target_seq[0, 0, sampled_token_index] = 1.
#         states_value = [h, c]

#     return decoded_sentence.strip()

# # ========== Routes ==========
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json['message']
#     input_matrix = preprocess_input(user_input)
#     response = decode_response(input_matrix)
#     return jsonify({'reply': response})

# # ========== Run Server ==========
# if __name__ == '__main__':
#     app.run(debug=True, port=8000)

# ------------------------------- MODEL 3 -------------------------------
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle, re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load files
model = load_model('chatbot_lstm_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
df = pd.read_csv('df.csv')

# Preprocess tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# TF-IDF Matrix
tfidf_matrix = vectorizer.transform(df['Clean_Questions'])

def sentence_case(text):
    sentences = re.split(r'([.!?])', text)
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    return ' '.join(sentences)

# Chatbot logic
def chatbot_response(user_input):
    user_input_clean = preprocess_text(user_input)
    user_vec = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    max_sim_idx = similarity.argmax()
    max_sim_score = similarity[0, max_sim_idx]

    if max_sim_score > 0.2:
        response = df.iloc[max_sim_idx]['Answers']
    else:
        response = "Maaf, saya belum bisa memahami pertanyaanmu. Coba ajukan dengan kata lain."

    # 🔽 Bersihkan token khusus LSTM jika ada
    response = response.replace('<start>', '').replace('<end>', '').strip()
    # 🔽 Ubah jadi sentence case
    response = sentence_case(response)
    return response


# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/get_response', methods=['POST'])
# def get_response():
#     user_input = request.json['message']
#     response = chatbot_response(user_input)
#     return jsonify({'response': response})

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['message']
    response = chatbot_response(user_input)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True, port=8000)

