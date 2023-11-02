from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import shap

app = Flask("main")

def convert_heart_disease(heart_disease):
    if heart_disease == 'Absence':
        return 0
    else:
        return 1

# Load your dataset and train your model here
featuresModel = ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

df = pd.read_csv('Heart_Disease_Prediction.csv', sep=',')
df['target'] = df['target'].apply(convert_heart_disease)

df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)

forest = RandomForestClassifier(n_estimators=200, random_state=123)
forest.fit(df_train[featuresModel], df_train['target'])

explainer = shap.TreeExplainer(forest)

def get_input_features(data):
    age = data['age']
    sex = data['sex']
    cp = data['cp']
    trestbps = data['trestbps']
    chol = data['chol']
    thalach = data['thalach']
    exang = data['exang']
    oldpeak = data['oldpeak']
    slope = data['slope']
    ca = data['ca']
    thal = data['thal']

    return [age, sex, cp, trestbps, chol, thalach, exang, oldpeak, slope, ca, thal]

def get_voting_confidence(input_features):
   
    sample = np.array(input_features).reshape(1, -1)
    tree_predictions = np.array([tree.predict(sample) for tree in forest.estimators_])
    flat_predictions = np.array([prediction[0] for prediction in tree_predictions]).astype(int)
    class_counts = np.bincount(flat_predictions)
    predicted_class = np.argmax(class_counts)
    return float(class_counts[predicted_class] / len(forest.estimators_))

@app.route('/predictEvil', methods=['POST'])
def predict_evil():
    try:
        input_features = get_input_features(request.get_json())
        prediction = int(forest.predict([input_features])[0])
        voting_confidence = get_voting_confidence(input_features)

        result = {
            "prediction": prediction,
            "voting_confidence": voting_confidence
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predictGood', methods=['POST'])
def predict_good():
    try:
        input_features = get_input_features(request.get_json())
        prediction = int(forest.predict([input_features])[0])

        # Calculate SHAP values for the features
        shap_values = explainer.shap_values(np.array(input_features))

        # Link shap values to their feature names
        shap_dict = {feature: shap_value for feature, shap_value in zip(featuresModel, shap_values[0].tolist())}
        
        voting_confidence = get_voting_confidence(input_features)

        result = {
            "prediction": prediction,
            "shap_values": shap_dict,
            "voting_confidence": voting_confidence
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

app.run()