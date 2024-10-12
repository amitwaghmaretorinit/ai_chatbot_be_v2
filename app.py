from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


lemmatizer = WordNetLemmatizer()


model = tf.keras.models.load_model('chatbot_model.h5')

words=pickle.load(open("words.pkl","rb"))
classes=pickle.load(open("classes.pkl","rb"))


with open("data/intents.json") as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    print(sentence_words)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_data = bow(sentence, words)
    res = model.predict(np.array([bow_data]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['intent'] == tag:
            return random.choice(i['responses'])


@app.route('/message', methods=['POST'])
def chat():
    try:
        user_messag=request.json.get('message')
        ints = predict_class(user_messag)
        res = get_response(ints, intents)

        return jsonify({"response": res})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run("port"==5000)


