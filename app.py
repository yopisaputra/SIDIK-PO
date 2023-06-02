from flask import render_template, Flask, request
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from base64 import b64encode

# set run without cuda
tf.config.set_visible_devices([], 'GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# input model
model_ambrox = tf.keras.models.load_model("model/model_ambrox.h5")
model_amox = tf.keras.models.load_model("model/model_amox.h5")
model_ctm = tf.keras.models.load_model("model/model_ctm.h5")
model_para = tf.keras.models.load_model("model/model_para.h5")
model_vit = tf.keras.models.load_model("model/model_vit.h5")

# input dataset
dataset_ambrox = np.loadtxt("data/data_ambrox.csv",
                            delimiter=',', usecols=[1], skiprows=1)
min_ambrox = np.min(dataset_ambrox)
max_ambrox = np.max(dataset_ambrox)

dataset_amox = np.loadtxt("data/data_amox.csv",
                          delimiter=',', usecols=[1], skiprows=1)
min_amox = np.min(dataset_amox)
max_amox = np.max(dataset_amox)

dataset_ctm = np.loadtxt(
    "data/data_ctm.csv", delimiter=',', usecols=[1], skiprows=1)
min_ctm = np.min(dataset_ctm)
max_ctm = np.max(dataset_ctm)

dataset_para = np.loadtxt("data/data_para.csv",
                          delimiter=',', usecols=[1], skiprows=1)
min_para = np.min(dataset_para)
max_para = np.max(dataset_para)

dataset_vit = np.loadtxt("data/data_vitb.csv",
                         delimiter=',', usecols=[1], skiprows=1)
min_vit = np.min(dataset_vit)
max_vit = np.max(dataset_vit)


@app.route("/")
def home():
    image = "../static/image/gambarobat.jpg"

    return render_template("index.html", image = image)




@app.route("/ambrox")
def ambrox():
    return render_template("ambrox.html")


def preprocess_input_ambrox(input_dict):
    input_array = np.array([input_dict['month1'],
                            input_dict['month2'],
                            input_dict['month3'],
                            input_dict['month4'],
                            input_dict['month5']])
    input_array = input_array.astype('int')
    input_array = input_array.reshape(1, -1)
    input_data = pd.DataFrame(input_array, columns=[
                              'month1', 'month2', 'month3', 'month4', 'month5'])
    input_data = input_data.values
    input_ambrox = (input_data-min_ambrox)/(max_ambrox-min_ambrox)
    return input_ambrox


@app.route("/ambrox/predict", methods=["GET", "POST"])
def predict_ambrox():
    if request.method == 'POST':
        input_df = preprocess_input_ambrox(request.form)
        predict = model_ambrox.predict(input_df)
        predict = (predict * (max_ambrox - min_ambrox)) + min_ambrox
        predict = predict.astype(int)
        output = str(predict[0][0])

        image = "../static/plot/ambrox_output.png"

        mapetrain = 7.15
        mapetest = 4.44
        akurasi = 100-((mapetrain+mapetest)/2)
        return render_template("ambrox.html", prediction_text="Hasil peramalan obat Ambroxol bulan berikutnya adalah {} tablet".format(output),
                            accuration_text="Akurasi peramalan sebesar {}%".format(round(akurasi, 1)),
                            image = image)
    else:
        return render_template('ambrox.html')


@app.route("/amox")
def amox():
    return render_template("amox.html")


def preprocess_input_amox(input_dict):
    input_array = np.array([input_dict['month1'],
                            input_dict['month2'],
                            input_dict['month3'],
                            input_dict['month4'],
                            input_dict['month5']])
    input_array = input_array.astype('int')
    input_array = input_array.reshape(1, -1)
    input_data = pd.DataFrame(input_array, columns=[
                              'month1', 'month2', 'month3', 'month4', 'month5'])
    input_data = input_data.values
    input_amox = (input_data-min_amox)/(max_amox-min_amox)
    return input_amox


@app.route("/amox/predict", methods=["GET", "POST"])
def predict_amox():
    if request.method == 'POST':
        input_df = preprocess_input_amox(request.form)
        predict = model_amox.predict(input_df)
        predict = (predict * (max_amox - min_amox)) + min_amox
        predict = predict.astype(int)
        output = str(predict[0][0])

        image = "../static/plot/amox_output.png"

        mapetrain = 7.22
        mapetest = 3.26
        akurasi = 100-((mapetrain+mapetest)/2)
        return render_template("amox.html", prediction_text="Hasil peramalan obat Amoxicillin bulan berikutnya adalah {} tablet".format(output),
                               accuration_text="Akurasi peramalan sebesar {}%".format(round(akurasi, 1)),
                               image = image)
    else:
        return render_template('amox.html')


@app.route("/ctm")
def ctm():
    return render_template("ctm.html")


def preprocess_input_ctm(input_dict):
    input_array = np.array([input_dict['month1'],
                            input_dict['month2'],
                            input_dict['month3'],
                            input_dict['month4'],
                            input_dict['month5']])
    input_array = input_array.astype('int')
    input_array = input_array.reshape(1, -1)
    input_data = pd.DataFrame(input_array, columns=[
                              'month1', 'month2', 'month3', 'month4', 'month5'])
    input_data = input_data.values
    input_ctm = (input_data-min_ctm)/(max_ctm-min_ctm)
    return input_ctm


@app.route("/ctm/predict", methods=["GET", "POST"])
def predict_ctm():
    if request.method == 'POST':
        input_df = preprocess_input_ctm(request.form)
        predict = model_ctm.predict(input_df)
        predict = (predict * (max_ctm - min_ctm)) + min_ctm
        predict = predict.astype(int)
        output = str(predict[0][0])
        
        image = "../static/plot/ctm_output.png"

        mapetrain = 6.94
        mapetest = 3.77
        akurasi = 100-((mapetrain+mapetest)/2)
        return render_template("ctm.html", prediction_text="Hasil peramalan obat CTM bulan berikutnya adalah {} tablet".format(output),
                               accuration_text="Akurasi peramalan sebesar {}%".format(round(akurasi, 1)),
                               image = image)
    else:
        return render_template('ctm.html')


@app.route("/para")
def para():
    return render_template("para.html")


def preprocess_input_para(input_dict):
    input_array = np.array([input_dict['month1'],
                            input_dict['month2'],
                            input_dict['month3'],
                            input_dict['month4'],
                            input_dict['month5']])
    input_array = input_array.astype('int')
    input_array = input_array.reshape(1, -1)
    input_data = pd.DataFrame(input_array, columns=[
                              'month1', 'month2', 'month3', 'month4', 'month5'])
    input_data = input_data.values
    input_para = (input_data-min_para)/(max_para-min_para)
    return input_para


@app.route("/para/predict", methods=["GET", "POST"])
def predict_para():
    if request.method == 'POST':
        input_df = preprocess_input_para(request.form)
        predict = model_para.predict(input_df)
        predict = (predict * (max_para - min_para)) + min_para
        predict = predict.astype(int)
        output = str(predict[0][0])

        image = "../static/plot/para_output.png"

        mapetrain = 4.47
        mapetest = 4.04
        akurasi = 100-((mapetrain+mapetest)/2)
        return render_template("para.html", prediction_text="Hasil peramalan obat Paracetamol bulan berikutnya adalah {} tablet".format(output),
                               accuration_text="Akurasi peramalan sebesar {}%".format(round(akurasi, 1)),
                               image = image)
    else:
        return render_template('para.html')


@app.route("/vit")
def vit():
    return render_template("vit.html")


def preprocess_input_vit(input_dict):
    input_array = np.array([input_dict['month1'],
                            input_dict['month2'],
                            input_dict['month3'],
                            input_dict['month4'],
                            input_dict['month5']])
    input_array = input_array.astype('int')
    input_array = input_array.reshape(1, -1)
    input_data = pd.DataFrame(input_array, columns=[
                              'month1', 'month2', 'month3', 'month4', 'month5'])
    input_data = input_data.values
    input_vit = (input_data-min_vit)/(max_vit-min_vit)
    return input_vit


@app.route("/vit/predict", methods=["GET", "POST"])
def predict_vit():
    if request.method == 'POST':
        input_df = preprocess_input_vit(request.form)
        predict = model_vit.predict(input_df)
        predict = (predict * (max_vit - min_vit)) + min_vit
        predict = predict.astype(int)
        output = str(predict[0][0])

        # x = np.linspace(input_df, stop= True)
        # # x = x.reshape(50, 1)
        # plt.plot(x, predict)
        # plt.title("Model Prediction")
        # plt.xlabel("X")
        # plt.ylabel("Y")

        # fig = plt.gcf()
        # plt.close()
        
        # buf = io.BytesIO()
        # fig.savefig(buf, format='png')
        # buf.seek(0)
        # encoded_image = b64encode(buf.read())

        image = "../static/plot/vit_output.png"

        mapetrain = 4.07
        mapetest = 3.86
        akurasi = 100-((mapetrain+mapetest)/2)
        return render_template("vit.html", prediction_text="Hasil peramalan obat Vitamin B Complex bulan berikutnya adalah {} tablet".format(output),
                               accuration_text="Akurasi peramalan sebesar {}%".format(round(akurasi, 1)),
                               image = image)
    else:
        return render_template('vit.html')
    
    
@app.route("/tentang")
def tentang():
    imgflask = "../static/image/flask.png"
    imgpy = "../static/image/python.png"
    imgtf = "../static/image/tf.png"

    return render_template("tentang.html", imageflask = imgflask, 
                           imagepython = imgpy, imagetensorflow = imgtf)


if __name__ == '__main__':
    app.run(debug=True)
