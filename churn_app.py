import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load

from tensorflow.keras.models import load_model
app = Flask(__name__)

model = load_model("classification.h5")
sc=load("transform1.save")

@app.route('/')
def home():
    return render_template('bank.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    z = request.form['Geo']
    if (z == "France"):
        o1,o2,o3 = 1,0,0
    if (z == "Germany"):
        o1,o2,o3 = 0,1,0
    if (z == "Spain"):
        o1,o2,o3 = 0,0,1
    a = request.form['CreditScore']
    b = request.form['Gen']
    if(b=='Female'):
        l1=0
    if(b=='Male'):
    	l1 = 1
    c = request.form['Age']
    d = request.form['Tenure']
    e = request.form['Balance']
    f = request.form['NumOfProducts']
    g = request.form['HasCrCard']
    h = request.form['IsActiveMember']
    i = request.form['EstimatedSalary']
    total = [[o1,o2,o3,a,l1,c,d,e,f,g,h,i]]
    prediction = model.predict(sc.transform(total))
    
    if(prediction==0):
        output = "He|She will stay in Bank"
    else:
        output="He|She will exit from Bank"

    return render_template('bank.html', prediction_text='Result: {}'.format(output))

if __name__ == "__main__":
    app.run(port=8086,debug=True)
