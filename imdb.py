import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
import re
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from tensorflow.keras.models import load_model
app = Flask(__name__)

model = load_model("imdb.h5")
sc=load("transform1.save")

@app.route('/')
def home():
    return render_template('imdb_app.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    
    r=request.form['review']
    r=r.lower()
    r=re.sub("[^a-zA-Z]", " ",str(r))
    r=r.split()
    r=[ps.stem(word) for word in r if not word in set(stopwords.words('english'))]
    r=' '.join(r)
    
    cv=CountVectorizer(max_features=2000)
    prediction = model.predict(cv.fit_transform([[r]]))
    
    if(prediction>0.5):
        output = "This is a positive comment"
    else:
        output="This is a negative comment"

    return render_template('imdb_app.html', prediction_text='Result: {}'.format(output))

if __name__ == "__main__":
    app.run(port=8086,debug=True)
