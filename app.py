from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import os
from wsgiref import simple_server 
import json
import re
import numpy as np
import pandas as pd
from tabula import read_pdf
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
from rich.progress import track
from matplotlib import pyplot as plt
#from werkzeug.datastructure import FileStorage
from werkzeug.utils import secure_filename
import seaborn as sns
app = Flask(__name__) 


def get_classes(value):
    classes ={
        0: 'Authentic',
        1: 'Fraudlent'
    }
    return classes[value]

class Preprocessing:

    def __init__(self) -> None:
        pass

    def preprocess(self, path):
        dataframe = read_pdf(path, pages="all")
        columns = ['Line No.','Posting Date', 'Due Date', 'Document No.', 'Trans. No.', 'Account','Debit/Credit']
        df = pd.DataFrame(dataframe[0].values, columns=columns)
        for i in track(range(1, len(dataframe))):
            df = df.append(pd.DataFrame(dataframe[i].values, columns=columns),ignore_index=True,)
            df = df.append(pd.DataFrame([dataframe[i].columns.to_list()], columns=columns),ignore_index=True,)
        df.replace(np.nan,0, inplace=True)
        df['Line No.'] = df['Line No.'].astype(int)
        df['Trans. No.'] = df['Trans. No.'].astype(int)
        return df

    def str_to_ammount(self, value):
        if type(value) == str:
            # should not remove - sign because it is used to get debit and credit
            value = re.sub(r'[^\d.-]', '', value)
            if value == '':
                return 0
            if value == '-':
                return 0
            value = float(value)
        if type(value) == float:
            return value
        if value == '':
            return 0
        return float(value)

    def get_debit(self, value):
        if value < 0:
            return value
        return 0
    
    def get_credit(self, value):
        if value > 0:
            return value
        return 0
    
    def get_classes(self, value):
        classes ={
            0: 'Authentic',
            1: 'Fraudlent'
        }
        return classes[value]
    
    def GetAccountType(self,dataframe):
        dataframe['Account']= dataframe['Account'].astype(str)
        for i in track(dataframe['Account'].index):
            if dataframe['Debit'][i] == 0:
                dataframe['Account'][i] = dataframe['Account'][i] + '_C'
            if dataframe['Credit'][i] == 0:
                dataframe['Account'][i] = dataframe['Account'][i] + '_D'
        return dataframe

le= pickle.load(open('models\label_encoder.pkl','rb'))
scaler=pickle.load(open('models\scaler.pkl','rb'))
svm= pickle.load(open('models\SVC.pkl','rb'))

@app.route('/')  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
        return redirect(url_for('index'))
    return render_template('index.html')


@app.route("/predict",methods=['POST','GET'])
def predict():
    # Read the fil
    # e
    #'Account', 'Debit', 'Credit', 'Label']
    #contents="data\dataset.pdf"
    # Preprocess the file
    if request.method == 'POST':
        f = request.files['myfile']
        f.save(secure_filename(f.filename))
    print('file uploaded successfully')
    contents=f
    preprcs = Preprocessing()
    dataframe = preprcs.preprocess(contents)
    dataframe['Debit/Credit'] = dataframe['Debit/Credit'].apply(preprcs.str_to_ammount)
    dataframe['Debit'] = dataframe['Debit/Credit'].apply(preprcs.get_debit)
    dataframe['Credit'] = dataframe['Debit/Credit'].apply(preprcs.get_credit)
    # Predict the file
    # Return the prediction
    #print(.head().to_string())
    df=dataframe
    df.drop(['Line No.', 'Posting Date','Due Date', 'Document No.', 'Trans. No.','Debit/Credit'],axis=1,inplace=True)
    df=preprcs.GetAccountType(df)
    df['Account']=le.transform(df['Account'])
    df=scaler.transform(df)
    pred=svm.predict(df)
    o=[get_classes(i) for i in pred ]
    print(o[0])
    #return str(o)
    # sns.countplot(x ='o', data = o)
    # plt.show()

if __name__ == "__main__":
    app.run(debug=True,port=8080 ,host='0.0.0.0')