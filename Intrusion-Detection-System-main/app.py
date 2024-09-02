import io
from flask import Flask, request, jsonify, render_template,make_response
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
import itertools
import traceback
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)
model_path = 'model.pkl'
label_encoders={}
@app.route('/train', methods=['POST'])
def train():
    global label_encoders
    csv_data = request.data.decode('utf-8')
    data = pd.read_csv(io.StringIO(csv_data))

    if 'class' not in data.columns:
        return ({'error': 'No target column named "class" in the dataset'}), 400
        #return make_response(('error': 'No target column named "class" in the dataset'), 200)

    data = data.drop_duplicates()
    missing_columns = [col for col in data.columns if data[col].isnull().sum() > 0]
    for col in missing_columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    for col in data.columns:
        if data[col].dtype == 'object':
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])


    if 'num_outbound_cmds' in data.columns:
        data.drop(['num_outbound_cmds'], axis=1, inplace=True)

    X = data.drop(['class'], axis=1)
    y = data['class']

    rfc = RandomForestClassifier()
    rfe = RFE(rfc, n_features_to_select=10)
    rfe = rfe.fit(X, y)
    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X.columns)]
    selected_features = [v for i, v in feature_map if i]
    X = X[selected_features]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    lg_model = LogisticRegression(random_state=55)
    lg_model.fit(X, y)

    joblib.dump((lg_model,scaler,label_encoders,selected_features), model_path)
    print("Training completed and selected_features:", selected_features)
    '''node_js_url = 'http://localhost:8000/predict'
    response= requests.post(node_js_url, json={'selected_features': selected_features})

    if response.ok:
       return jsonify({'message': 'Selected features sent to Node.js server successfully'})
    else:
        return jsonify({'error': 'Failed to send selected features to Node.js server'}), 500'''

    '''loaded_model = joblib.load(model_path)

    # Check the contents of the loaded model
    if isinstance(loaded_model, tuple) and len(loaded_model) == 3:
        lg_model, scaler, selected_features = loaded_model
        print("Selected features:", selected_features)
    else:
        print("Model file does not contain the expected contents.")'''
    return jsonify({
        'message': 'Model trained successfully'
    })

'''@app.route('/selected_features', methods=['GET'])
def get_selected_features():
    model, _, selected_features = joblib.load(model_path)
    return jsonify({'selected_features': selected_features})'''
    
'''@app.route('/predict', methods=['POST'])
def predict():
    model, scaler, selected_features = joblib.load(model_path)
    input_data = request.json
    input_df = pd.DataFrame([input_data], columns=selected_features)
    input_df = scaler.transform(input_df)
    prediction = model.predict(input_df)[0]
    return jsonify({'prediction': 'normal' if prediction == 0 else 'anomaly'})

@app.route('/predict.html', methods=['GET'])
def predict_page():
    model, _, selected_features = joblib.load(model_path)
    return render_template('predict.html', features=selected_features)'''
@app.route('/')
def index():
    # Load the selected features from the model file
    model, scaler, label_encoder,selected_features = joblib.load(model_path)
    return render_template('predict.html', selected_features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model, scaler, label_encoders, selected_features = joblib.load(model_path)

        input_data = request.json
        logging.debug(f"Received input data: {input_data}")

        input_df = pd.DataFrame([input_data], columns=selected_features)
        logging.debug(f"Input DataFrame: {input_df}")

        logging.debug(f"Label Encoder Classes: {label_encoders}")

        for col in input_df.columns:
            print(input_df[col].dtype)
            if input_df[col].dtype == 'object':
                if col in label_encoders:
                    label_encoder = label_encoders[col]
                    input_df[col] = label_encoder.transform(input_df[col])
                else:
                    logging.error(f"Label encoder not found for column '{col}'")
                    return jsonify({'error': f'Label encoder not found for column "{col}"'}), 400

        input_df = scaler.transform(input_df)
        logging.debug(f"Input DataFrame after scaling: {input_df}")

        prediction = model.predict(input_df)[0]
        prediction_label = 'normal' if prediction == 0 else 'anomaly'
        logging.debug(f"Prediction: {prediction_label}")

        return jsonify({'prediction': prediction_label})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred during prediction'}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
