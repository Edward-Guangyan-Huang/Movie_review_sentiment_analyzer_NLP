import numpy as np
from flask import Flask, request,render_template,jsonify
import pickle

app = Flask(__name__)
naive_bayes2 = pickle.load(open('naive_bayes2.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    new_comment = [x for x in request.form.values()]
    new_comment = str(new_comment)
    new_comment = [new_comment]
    new_comment = vectorizer.transform(new_comment)
    new_comment = new_comment.toarray()
    pred_sentiment = naive_bayes2.predict(new_comment)

    if pred_sentiment == 'pos':
        return render_template('index.html', prediction_text = 'This review is a positive one.')

    elif pred_sentiment == 'neg':
        return render_template('index.html', prediction_text = 'This review is a negative one.')
    
    else:
        return render_template('index.html', prediction_text = 'This review is a mixed one.')

if __name__ == "__main__":
    app.run(debug=True)
