
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)

sess=tf.Session() #default a new session for each request
graph=tf.get_default_graph() 
set_session(sess) 
model = load_model("fruit.h5")

                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64)) 
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x,axis =0)
        print(x)
        global sess
        global graph 
        with graph.as_default():
                set_session(sess)
                preds = model.predict(x)
        #keras.backend.clear_session()
        #preds = model.make_predict_function(x)
        print("prediction",preds)
        index = ['Apple','Banana','Lemon','Orange','Pear']
        text = "The classified fruit is : " + index[np.argmax(preds[0])]
    return text
if __name__ == '__main__':
    #graph = tf.get_default_graph()
	#model = keras.models.load_model('./data/model/model.model')
    app.run(debug = True, threaded = False)
