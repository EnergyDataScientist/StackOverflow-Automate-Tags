#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 18:05:28 2021

@author: hugo
"""

# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, jsonify
import pickle

import pandas as pd
import numpy as np
import pickle

# text preprocessing
import re
from string import punctuation
import nltk
from bs4 import BeautifulSoup
import spacy

# sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_score
from sklearn.pipeline import Pipeline

nlp = spacy.load("en_core_web_sm")
#nltk.download()
n_jobs = -1 # enable multiprocessing

df = pd.read_csv('df_process.csv', sep=';')

df['Tags_process'] = df['Tags_process'].apply(lambda x: [text[1:-1] for text in x.strip('[]').split(', ')])
df['Tags'] = df['Tags'].apply(lambda x: [text[1:-1] for text in x.strip('[]').split(', ')])

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





app = Flask(__name__)

# Loading the multilabel binazer
mlb_pkl = open('mlb.pkl', 'rb')
mlb = pickle.load(mlb_pkl)
print(f'Loaded multilabel binarizer : {mlb}')

# Loading the pipeline
best_pipeline_pkl = open('best_pipeline.pkl', 'rb')
best_pipeline = pickle.load(best_pipeline_pkl)
print(f'Loaded best pipeline : {best_pipeline}')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data_html = request.form['post']
    final_features = [data_html]
    
    prediction_normalised = best_pipeline.predict(final_features)
    prediction_index = prediction_normalised == 1
    
    prediction = mlb.classes_[prediction_index.ravel()]

    return render_template('index.html', prediction_text='Tags : {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)

