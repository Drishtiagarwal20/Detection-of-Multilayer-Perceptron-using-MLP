from flask import Flask, render_template, request
from flask.helpers import make_response
import numpy as np
import tensorflow
model = tensorflow.keras.models.load_model('trained_model')

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('first.html') 

database={'admin':'admin'}

@app.route('/print', methods=['POST'])
def print():
    n = request.form['name']
    p = request.form['cn']
    g = request.form.get('gender')
    d1 = float(request.form['a'])
    d2 = float(request.form['b'])
    d3 = float(request.form['c'])
    d4 = float(request.form['d'])
    d5 = float(request.form['e'])
    d6 = float(request.form['f'])
    d7 = float(request.form['g'])
    d8 = float(request.form['h'])
    d9 = float(request.form['i'])
    d10 = float(request.form['j'])
    d11 = float(request.form['k'])
    d12 = float(request.form['l'])
    d13 = float(request.form['m'])
    d14 = float(request.form['n'])
    d15 = float(request.form['o'])
    arr = np.array([[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15]])
    pred = model.predict(arr)
    app.logger.info(arr)
    return render_template('pdf.html', n=n,p=p,g=g,d1=d1,d2=d2,d3=d3,d4=d4,d5=d5,d6=d6,d7=d7,d8=d8,d9=d9,d10=d10,d11=d11,d12=d12,d13=d13,d14=d14,d15=d15,data=pred)

@app.route('/go')
def go():
    return render_template('login.html')

@app.route('/exit')
def exit():
    return render_template('home.html')

@app.route('/form_login',methods=['POST'])
def login():
    n= request.form['un']
    p= request.form['pass']
    
    if n not in database:
        return render_template('login.html',info="Invalid Credentials")
    else:
        if database[n]!=p:
            return render_template('login.html',info="Invalid Password")
        else:
            return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    d1 = float(request.form['a'])
    d2 = float(request.form['b'])
    d3 = float(request.form['c'])
    d4 = float(request.form['d'])
    d5 = float(request.form['e'])
    d6 = float(request.form['f'])
    d7 = float(request.form['g'])
    d8 = float(request.form['h'])
    d9 = float(request.form['i'])
    d10 = float(request.form['j'])
    d11 = float(request.form['k'])
    d12 = float(request.form['l'])
    d13 = float(request.form['m'])
    d14 = float(request.form['n'])
    d15 = float(request.form['o'])
    arr = np.array([[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

if __name__=="__main__":
    app.run(debug=True)
