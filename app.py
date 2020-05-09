import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.python.keras.preprocessing import text, sequence
import tensorflow as tf
import copy
import os
import random
from tokenizer import x_tokenizer


app = Flask(__name__,static_url_path='/static')
#model = pickle.load(open('model.pkl', 'rb'))
modelSR = tf.keras.models.load_model('modelSR.h5')

@app.route('/')
def home():
    return render_template('index.html')

'''
@app.route('/predict',methods=['POST'])

def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
'''

@app.route('/predictSR',methods=['POST'])

def predictSR():
    input = [request.form.get("user_input")]
    #input = ['My dog dead, I am so sad.']
    string_tokenized = x_tokenizer.texts_to_sequences(input)
    x_testing = sequence.pad_sequences(string_tokenized, maxlen=400)
    y_testing = modelSR.predict(x_testing, verbose = 1)

    list_of_classes = ['happiness','sadness','surprise','hate','love']
    y_class = list_of_classes[y_testing.argmax()]

    path="static/img"
    label_list = os.listdir(path)
    img_names = copy.deepcopy(label_list)

    for index,name in enumerate(label_list):
        label_list[index] = name.replace('.jpg', '')
        label_list[index] = label_list[index].split(',')

    dic = {img_names[i]: label_list[i] for i in range(len(img_names))}

    def find_key(input_dict, key_word):
        return [key for key, value in input_dict.items() if key_word in value]

    key_word = y_class
    stickers_name = find_key(dic,key_word)
    random_sticker_name = random.choice(stickers_name)

    return render_template('index.html', prediction_class = y_class, prediction_text = random_sticker_name)



@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
