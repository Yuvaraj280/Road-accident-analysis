from flask import Flask, request, render_template
import numpy as np
import pickle
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["CACHE_TYPE"] = "null"
# change to "redis" and restart to cache again

# some time later
file=open('my_model.pkl','rb')
model=pickle.load(file)
file.close()
      




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    #int_features = [int(x) for x in request.form.values()]
    if request.method == 'POST':
        NOV = request.form["NOV"]
        FRC = request.form["FRC"]
        RC = request.form["RC"]
        LC = request.form["LC"]
        WC = request.form["WC"]
        VN = request.form["VN"]
        TOV = request.form["TOV"]
        CC = request.form["CC"]
        SOC = request.form["SOC"]
        
        '''data= [NOV, FRC, RC, LC, WC, VN, TOV, CC, SOC, AOC]
        data = np.array(data)
        data = data.astype(np.float).reshape(1,-1)
        predict = model.predict(data) '''
        
        data= [NOV, FRC, RC, LC, WC, VN, TOV, CC, SOC]
        data = np.array(data)
        np.float = float 
        data = data.astype(np.float).reshape(1,-1)
        predict = model.predict(data)
        
        
        
        if(predict==1):
            score='Slight!'
        elif(predict<2.5):
            score='Serious!'
        elif(predict>=2.5):
            score='FATAL'
        else:
            score='Low' 


    return render_template('index.html', prediction_text='The Accident Severity is *{}* ,   Score : {} '.format(score,predict))

     
   


   

    #return render_template('index.html', prediction_text='The Accident Severity is *{}* ,   Score : {} / 4'.format(score,output))

"""@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
"""

@app.route('/dashboard')
def dashboard():
    
    return render_template('dashboard.html')


if __name__ == "__main__":
    app.run(debug=False)
