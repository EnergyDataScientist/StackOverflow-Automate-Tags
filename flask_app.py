#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 18:05:28 2021

@author: hugo
"""

# -*- coding: utf-8 -*-
from flask import Flask, request, render_template

# text preprocessing
import re
from string import punctuation
import nltk
from bs4 import BeautifulSoup
import spacy
import joblib

import os
### Loading model
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
mlb_path = os.path.join(THIS_FOLDER, 'mlb.pkl')
transformer_path = os.path.join(THIS_FOLDER, 'transformer.pkl')
model_path = os.path.join(THIS_FOLDER, 'model.pkl')

# Loading the multilabel binazer
mlb = joblib.load(mlb_path)
print(f'Loaded multilabel binarizer : {mlb}')

# Loading the transformer
transformer = joblib.load(transformer_path)
print(f'Loaded transformer : {transformer}')

# Loading the model
model = joblib.load(model_path)
print(f'Loaded model : {model}')

### Text prepocessing
nlp = spacy.load('en_core_web_sm')
n_jobs = -1 # enable multiprocessing

sw = nltk.corpus.stopwords.words('english')
sw.extend(['error', 'code', 'program', 'question', 'result'])
stemmer = nltk.stem.snowball.SnowballStemmer("english")

def removeTag(text):
    tag_list = ['code','a','img','kbd','del','strike','s']
    soup = BeautifulSoup(text, "html.parser")

    for tag in tag_list:
        for tagless in soup.find_all(tag):
            tagless.decompose()

    # to get lowercase text
    return soup.get_text().lower()

def removePunctuation(text):
    cleaned = re.sub('\n',r' ',text)
    # It is prefereable to replace punctuation char by white space to avoid creating new words
    translate_table = dict((ord(char), ' ') for char in punctuation)
    cleaned = cleaned.translate(translate_table)
    cleaned = re.sub(r'\s+', ' ',cleaned)

    return cleaned

def textPreprocessingString(text, allowed_postags=['NOUN']):
    doc = nlp(text)
    cleaned = " ".join([token.lemma_ for token in doc if ((token.pos_ in allowed_postags) and (token.text not in sw))])

    return cleaned

def textPreprocessing(text):
    text_notag = removeTag(text)
    text_nopunct = removePunctuation(text_notag)

    return textPreprocessingString(text_nopunct)


### Flask API
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data_html = request.form['post']
    data_process = textPreprocessing(data_html)
    data_vec = transformer.transform([data_process])
    
    prediction_normalised = model.predict(data_vec)
    prediction_index = prediction_normalised == 1
    prediction = mlb.classes_[prediction_index.ravel()]

    return render_template('index.html', prediction_text='Tags : {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)