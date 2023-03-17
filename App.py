# Working file . Model should be dumped using pickle commnad not joblib

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template

import pickle

App = Flask(__name__)

model=pickle.load(open('wineQualitypredict.pkl','rb'))

input_scaler=pickle.load(open('wineQuality_input.pkl','rb'))




@App.route('/')
def home():
    return render_template('index.html')   # Home page

@App.route('/predict', methods=['POST'])    # Prediction
def predict():

    '''

    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    inp_data = input_scaler.transform([int_features])
    
    feat=model.feature_importances_
    feat
    l=list(feat)
    l=pd.DataFrame(l,columns=['list'])

    bestfea=set(l.list)
    featurename=l[l.list==feat.max()].iloc[:,1:]

    prediction = model.predict(inp_data)      # 2-d array is reuqired as an input
    
    output = str(round(prediction[0],2)) + ", Best feature index number :"+str(featurename) +'th'+'is :' + str(feat.max())
#+ ", Radio - " + str(int_features[1])+", Newspaper - " + str(int_features[2])
   
    
    return render_template('index.html', prediction_text='Predicted Wine quality is:{}'.format(output))



if __name__ == "__main__":
    App.run(debug=True)
