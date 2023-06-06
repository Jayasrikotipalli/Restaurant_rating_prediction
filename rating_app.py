import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = pickle.load(open('ML_Model.pkl', 'rb'))

encoder = LabelEncoder()
model = model[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    online_order = request.form.get('online_order')
    book_table = request.form.get('book table')
    location = request.form.get('location')
    rest_type = request.form.get('rest_type')
    dish_liked = request.form.get('dish_liked')
    cuisines = request.form.get('cuisines')
    approx_cost = request.form.get('approx_cost(for two people)')
    reviews_list = request.form.get('reviews_list')
    menu_item = request.form.get('menu_item')
    listed_in = request.form.get('listed_in(type)')

    approx_cost = float(approx_cost)
    features = [online_order, book_table, location, rest_type, dish_liked, cuisines, reviews_list, menu_item, listed_in]
    final_features = []
    for feature in features:
        feature_encoded = encoder.transform([feature])[0] if feature in encoder.classes_ else -1
        final_features.append(feature_encoded)

    final_features.insert(6, approx_cost)
    prediction = model.predict([final_features])
    return "The rating for this restaurant might be: {}".format(prediction[0])


if __name__ == "__main__":
    train_data_path = r"D:/projects_/zomato/train_data.csv"

    data = pd.read_csv(train_data_path)
    cate = data.select_dtypes(include=['object']).columns.tolist()

    for col in cate:
        encoder.fit(data[col])
        data[col] = encoder.transform(data[col])

    app.run(debug=True)
