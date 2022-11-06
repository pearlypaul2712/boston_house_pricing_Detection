import pickle

from flask import Flask,request,app,jsonify,render_template

import numpy  as np
import pandas as pd

## starting point of the application
app=Flask(__name__)
## Load the model 
reg_model=pickle.load(open('regmodelsample.pkl','rb'))
scalar_transform=pickle.load(open('scaling_model.pkl','rb'))

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    ## whenever hit predict i will give the input as data 
    data=request.json['data']
    print(np.array(list(data.values())).reshape(1,-1))
    ## Converting to list 
    new_transformed_data=scalar_transform.transform(np.array(list(data.values())).reshape(1,-1))

    output=reg_model.predict(new_transformed_data)
    print(output[0])
    return(jsonify(output[0]))


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    user_input=scalar_transform.transform(np.array(data).reshape(1,-1))
    print(user_input)
    output_values=reg_model.predict(user_input)
    print(output_values)
    return render_template("home.html",prediction_text="The Boston house price prediction is {}".format(output_values))


if __name__=="__main__":
    app.run(debug=True)






