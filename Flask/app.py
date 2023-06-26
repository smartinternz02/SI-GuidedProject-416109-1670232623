from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('CO2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction',methods=['POST','GET'])
def prediction(): # route which will take you to the prediction page
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI
def predict():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values() ]  
    features_values=[np.array(input_feature)]
    feature_name=['Make', 'Vehicle_Class', 'Engine_Size', 'Cylinders',
       'Transmission', 'Fuel_Type', 'Fuel_Consumption_City',
       'Fuel_Consumption_Hwy', 'Fuel_Consumption_Comb(mpg)']
    x=pd.DataFrame(features_values,columns=feature_name)
    
     # predictions using the loaded model file
    prediction=model.predict(x)  
    print("Prediction is:",prediction)
     # showing the prediction results in a UI
    return render_template("resultnew.html",prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)