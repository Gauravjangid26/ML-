from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import pickle 
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


scaler_log=pickle.load(open("/config/workspace/ML project on logistic regression/pickle/scaler_log (1).pkl","rb"))
clf=pickle.load(open("/config/workspace/ML project on logistic regression/pickle/clf (1).pickle","rb"))


@app.route('/')
def index():
    return render_template("index.html")
    
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        result=''

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose=float(request.form.get("Glucose"))
        BloodPressure=float(request.form.get("BloodPressure"))
        SkinThickness=float(request.form.get("SkinThickness"))
        Insulin=float(request.form.get("Insulin"))
        BMI=float(request.form.get("BMI"))
        DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction"))
        Age =float(request.form.get("Age"))
        
        new_data=scaler_log.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predicted=clf.predict(new_data)
        
        if predicted[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('single_prediction.html',result=result)


    else:
        return render_template("home.html")
if __name__=="__main__":
    app.run(host="0.0.0.0")
