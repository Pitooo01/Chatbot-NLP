from flask import Flask, render_template, request, jsonify
import numpy as np
import re
import os
import json
from keras.models import load_model, Model
from keras.layers import Input

app = Flask(__name__)

# ========== Load Model ==========
training_model = load_model('training_model.h5')

# Cek apakah file ada dan tidak kosong
if not os.path.exists('input_features_dict.json') or os.path.getsize('input_features_dict.json') == 0:
    raise ValueError("File input_features_dict.json tidak ditemukan atau kosong!")

# Load JSON
with open('input_features_dict.json') as f:
    input_features_dict = json.load(f)

# with open('input_features_dict.json') as f:
#     input_features_dict = json.load(f)

with open('target_features_dict.json') as f:
    target_features_dict = json.load(f)

with open('reverse_target_features_dict.json') as f:
    reverse_target_features_dict = json.load(f)

num_encoder_tokens = len(input_features_dict)
num_decoder_tokens = len(target_features_dict)
max_encoder_seq_length = 20  # sesuaikan dengan modelmu
max_decoder_seq_length = 20
latent_dim = 256

# ========== Build Encoder Model ==========
encoder_inputs = training_model.input[0]
_, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

# ========== Build Decoder Model ==========
decoder_inputs = training_model.input[1]
decoder_lstm = training_model.layers[3]
decoder_dense = training_model.layers[4]

# Tambahkan nama khusus untuk Input baru
decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Gunakan decoder_inputs dari model training sebelumnya
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_outputs = decoder_dense(decoder_outputs)

# Buat decoder_model dengan input yang sekarang namanya unik
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + [state_h, state_c])


# ========== Helper Function ==========
def preprocess_input(text):
    tokens = re.findall(r"[\w']+|[^\s\w]", text)
    input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
    for t, token in enumerate(tokens):
        if token in input_features_dict:
            input_matrix[0, t, input_features_dict[token]] = 1.
    return input_matrix

def decode_response(test_input):
    states_value = encoder_model.predict(test_input)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_features_dict['<START>']] = 1.

    decoded_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[str(sampled_token_index)]

        if sampled_token == '<END>' or len(decoded_sentence.split()) > max_decoder_seq_length:
            break

        decoded_sentence += ' ' + sampled_token
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]

    return decoded_sentence.strip()

# ========== Routes ==========
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    input_matrix = preprocess_input(user_input)
    response = decode_response(input_matrix)
    return jsonify({'reply': response})

# ========== Run Server ==========
if __name__ == '__main__':
    app.run(debug=True)
