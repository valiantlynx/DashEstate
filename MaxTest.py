import json
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load data from 'houses.jsonl' to extract unique values and maximums for the HTML page
data_houses = load_jsonl('data/houses.jsonl')
data_districts = load_jsonl('data/districts.jsonl')
data_agents = load_jsonl('data/agents.jsonl')
data_schools = load_jsonl('data/schools.jsonl')

# Extract unique district ID values for the dropdown
districts = {item['id'] for item in data_districts}

# Initialize unique values dict
unique_values = {
    "colors": list(set(item.get("color", "unknown") for item in data_houses)),
    "district_ids": sorted(list(districts)),
    "ratings": list(range(6)),
    "condition_ratings": list(range(11))
}

# Calculate maximum values for numeric inputs
maximums = {
    "year": max(item.get("year", 0) for item in data_houses if item.get("year") is not None),
    "remodeled": max(item.get("remodeled", 0) for item in data_houses if item.get("remodeled") is not None),
    "bathrooms": max(item.get("bathrooms", 0) for item in data_houses if item.get("bathrooms") is not None),
    "days_on_marked": max(item.get("days_on_marked", 0) for item in data_houses if item.get("days_on_marked") is not None),
    "external_storage_m2": max(
        item.get("external_storage_m2", 0) for item in data_houses if item.get("external_storage_m2") is not None),
    "kitchens": max(item.get("kitchens", 0) for item in data_houses if item.get("kitchens") is not None),
    "lot_w": max(item.get("lot_w", 0) for item in data_houses if item.get("lot_w") is not None),
    "price": max(item.get("price", 0) for item in data_houses if item.get("price") is not None),
    "rooms": max(item.get("rooms", 0) for item in data_houses if item.get("rooms") is not None),
    "storage_rating": max(item.get("storage_rating", 0) for item in data_houses if item.get("storage_rating") is not None),
    "sun_factor": max(item.get("sun_factor", 0) for item in data_houses if item.get("sun_factor") is not None),
}

def impute_missing_values(houses):
    numerical_features = [
        'size', 'bathrooms', 'condition_rating',
        'crime_rating', 'public_transport_rating',
        'school_rating', 'house_age'
    ]
    for feature in numerical_features:
        values = [house[feature] for house in houses if
                  house.get(feature) is not None and isinstance(house[feature], (int, float))]
        if values:
            mean_value = np.mean(values)
            for house in houses:
                if house.get(feature) is None or not isinstance(house.get(feature), (int, float)):
                    house[feature] = mean_value
        else:
            for house in houses:
                house[feature] = 0

def encode_categorical(houses):
    advertisement_types = {'regular': 0, 'premium': 1}
    for house in houses:
        house['advertisement'] = advertisement_types.get(house.get('advertisement', 'regular'), 0)

def add_house_age(houses, current_year):
    for house in houses:
        if 'year' in house and house['year'] is not None and isinstance(house['year'], (int, float)):
            house['house_age'] = current_year - house['year']
        else:
            house['house_age'] = 0

def prepare_dataset(houses):
    X = []
    y = []
    for house in houses:
        features = [
            house['advertisement'],
            house['bathrooms'],
            house['condition_rating'],
            house['size'],
            house['house_age'],
            house['crime_rating'],
            house['public_transport_rating'],
            house['school_rating']
        ]
        X.append(features)
        y.append(house['price'])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def train_test_split_custom(X, y, test_size=0.2):
    data_size = len(X)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    test_set_size = int(data_size * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

def initialize_parameters(n_features):
    w = np.zeros(n_features)
    b = 0
    return w, b

def predict(X, w, b):
    return np.dot(X, w) + b

def compute_loss(y_true, y_pred):
    m = len(y_true)
    residuals = y_pred - y_true
    if np.isnan(residuals).any() or np.isinf(residuals).any():
        print("Invalid residuals encountered.")
        return np.nan
    loss = (1 / (2 * m)) * np.sum(residuals ** 2)
    return loss

def compute_gradients_batch(X, y_true, y_pred):
    m = len(y_true)
    residuals = y_pred - y_true
    dw = (1/m) * np.dot(X.T, residuals)
    db = (1/m) * np.sum(residuals)
    return dw, db

def update_parameters(w, b, dw, db, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

def save_state_to_file(best_w, best_b, best_loss, file_path):
    state = {
        'weights': best_w.tolist(),
        'bias': best_b,
        'best_loss': best_loss
    }
    with open(file_path, 'w') as f:
        json.dump(state, f)

def load_state_from_file(file_path):
    with open(file_path, 'r') as f:
        state = json.load(f)
    best_w = np.array(state['weights'])
    best_b = state['bias']
    best_loss = state['best_loss']
    print(f"State loaded from {file_path}")
    return best_w, best_b, best_loss

def train_linear_regression(X_train, y_train_scaled, learning_rate=0.001, epochs=2000, batch_size=32, state_file='model_state.json'):
    n_samples, n_features = X_train.shape

    # Delete existing model files to ensure retraining with new feature_order
    if os.path.exists(state_file):
        os.remove(state_file)
    if os.path.exists('linear_regression_model.pkl'):
        os.remove('linear_regression_model.pkl')

    best_w, best_b = initialize_parameters(n_features)
    best_loss = float('inf')

    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_scaled[indices]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            y_pred_batch = predict(X_batch, best_w, best_b)
            loss = compute_loss(y_batch, y_pred_batch)
            if np.isnan(loss) or np.isinf(loss):
                print(f"Training stopped due to NaN or Inf loss at epoch {epoch}, batch starting at {start_idx}.")
                save_state_to_file(best_w, best_b, best_loss, state_file)
                return best_w, best_b, best_loss

            dw, db = compute_gradients_batch(X_batch, y_batch, y_pred_batch)
            if np.isnan(dw).any() or np.isnan(db) or np.isinf(dw).any() or np.isinf(db):
                print(f"Invalid gradients at epoch {epoch}, batch starting at {start_idx}.")
                save_state_to_file(best_w, best_b, best_loss, state_file)
                return best_w, best_b, best_loss

            best_w, best_b = update_parameters(best_w, best_b, dw, db, learning_rate)

        # Compute the loss over the entire training set for monitoring
        y_pred = predict(X_train, best_w, best_b)
        total_loss = compute_loss(y_train_scaled, y_pred)

        if total_loss < best_loss:
            best_loss = total_loss
            save_state_to_file(best_w, best_b, best_loss, state_file)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss}, Best Loss: {best_loss}')

    return best_w, best_b, best_loss

def evaluate_model(X_test, y_test_scaled, w, b):
    y_pred_scaled = predict(X_test, w, b)
    mse = compute_loss(y_test_scaled, y_pred_scaled) * 2
    rmse = np.sqrt(mse)
    ss_total = np.sum((y_test_scaled - np.mean(y_test_scaled)) ** 2)
    ss_residual = np.sum((y_test_scaled - y_pred_scaled) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    print(f'Test MSE: {mse}')
    print(f'Test RMSE: {rmse}')
    print(f'R-squared: {r_squared}')

def save_model(w, b, scaler_X, scaler_y, feature_order, filename='linear_regression_model.pkl'):
    model = {
        'weights': w,
        'bias': b,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_order': feature_order
    }
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename='linear_regression_model.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model['weights'], model['bias'], model['scaler_X'], model['scaler_y'], model['feature_order']

def preprocess_user_input(house_data, scaler_X, feature_order):
    processed_data = []
    for feature in feature_order:
        if feature == 'advertisement':
            advertisement_types = {'regular': 0, 'premium': 1}
            value = advertisement_types.get(house_data.get('advertisement', 'regular'), 0)
        else:
            value = house_data.get(feature, 0)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0
        processed_data.append(value)
    # Scale the features
    processed_data = np.array(processed_data).reshape(1, -1)
    processed_data_scaled = scaler_X.transform(processed_data)
    return processed_data_scaled.flatten()

@app.route('/')
def index():
    return render_template('index.html', unique_values=unique_values, maximums=maximums)

@app.route('/predict', methods=['POST'])
def predict_route():
    user_input = request.get_json()
    w, b, scaler_X, scaler_y, feature_order = load_model()

    model_input = {}
    for feature in feature_order:
        if feature == 'house_age':
            try:
                year = int(user_input.get('year', 2023))
                value = 2023 - year
            except (ValueError, TypeError):
                value = 0
        elif feature == 'advertisement':
            value = 'premium' if user_input.get('advertisement', False) else 'regular'
        else:
            value = user_input.get(feature, 0)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0
        model_input[feature] = value

    # Encode categorical features
    encode_categorical([model_input])

    # Preprocess input
    features_scaled = preprocess_user_input(model_input, scaler_X, feature_order)
    predicted_price_scaled = predict(features_scaled.reshape(1, -1), w, b)

    predicted_price = scaler_y.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]
    rounded_price = round(predicted_price, 2)
    return jsonify({'price': rounded_price}), 200

@app.route('/about')
def about():
    return "This is a test Flask app for DashEstate."

@app.route('/routes')
def list_routes():
    route_list = []
    for rule in app.url_map.iter_rules():
        route_info = {
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "rule": rule.rule
        }
        route_list.append(route_info)
    return jsonify(route_list), 200
if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        # Put your training code here
        agents = load_jsonl('data/agents.jsonl')
        districts = load_jsonl('data/districts.jsonl')
        schools = load_jsonl('data/schools.jsonl')
        houses = load_jsonl('data/houses.jsonl')

        agents_dict = {agent['agent_id']: agent for agent in agents}
        districts_dict = {district['id']: district for district in districts}
        schools_dict = {school['id']: school for school in schools}

        for house in houses:
            agent_info = agents_dict.get(house.get('agent_id'), {})
            house['agent_name'] = agent_info.get('name', 'Unknown')
            district_info = districts_dict.get(house.get('district_id'), {})
            house['crime_rating'] = district_info.get('crime_rating', None)
            house['public_transport_rating'] = district_info.get('public_transport_rating', None)
            school_info = schools_dict.get(house.get('school_id'), {})
            house['school_rating'] = school_info.get('rating', None)
            house['school_capacity'] = school_info.get('capacity', None)
            house['school_built_year'] = school_info.get('built_year', None)

        add_house_age(houses, current_year=2023)
        impute_missing_values(houses)
        encode_categorical(houses)

        feature_order = [
            'advertisement',
            'bathrooms',
            'condition_rating',
            'size',
            'house_age',
            'crime_rating',
            'public_transport_rating',
            'school_rating'
        ]

        X, y = prepare_dataset(houses)

        if np.isnan(X).any() or np.isinf(X).any():
            print("NaN or infinite values detected in feature matrix X.")
        if np.isnan(y).any() or np.isinf(y).any():
            print("NaN or infinite values detected in target vector y.")

        X_train, X_test, y_train, y_test = train_test_split_custom(X, y)

        # Scale features and target variable
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        # Train the model
        w, b, best_loss = train_linear_regression(X_train_scaled, y_train_scaled, learning_rate=0.001, epochs=2000,
                                                  batch_size=32)

        print(f'Best Loss: {best_loss}')
        print('Learned Weights:', w)
        print('Bias:', b)

        # Evaluate the model
        evaluate_model(X_test_scaled, y_test_scaled, w, b)

        # Save the model
        save_model(w, b, scaler_X, scaler_y, feature_order)

    app.run(debug=True)
