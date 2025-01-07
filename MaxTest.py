import json
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

data_houses = load_jsonl('data/houses.jsonl')
data_districts = load_jsonl('data/districts.jsonl')
data_agents = load_jsonl('data/agents.jsonl')
data_schools = load_jsonl('data/schools.jsonl')

# Create dictionaries for quick lookup
districts_dict = {district['id']: district for district in data_districts}
schools_dict = {school['id']: school for school in data_schools}

# Define unique values and maximums
unique_values = {
    "colors": list(set(item.get("color", "unknown") for item in data_houses)),
    "ratings": list(range(6)),
    "condition_ratings": list(range(11)),
    "advertisement_types": ["no", "premium", "regular", "unknown"]
}

maximums = {
    "year": max(item.get("year", 0) for item in data_houses if item.get("year") is not None),
    "remodeled": max(item.get("remodeled", 0) for item in data_houses if item.get("remodeled") is not None),
    "bathrooms": max(item.get("bathrooms", 0) for item in data_houses if item.get("bathrooms") is not None),
    "days_on_marked": max(
        item.get("days_on_marked", 0) for item in data_houses if item.get("days_on_marked") is not None),
    "external_storage_m2": max(
        item.get("external_storage_m2", 0) for item in data_houses if item.get("external_storage_m2") is not None),
    "kitchens": max(item.get("kitchens", 0) for item in data_houses if item.get("kitchens") is not None),
    "lot_w": max(item.get("lot_w", 0) for item in data_houses if item.get("lot_w") is not None),
    "price": max(item.get("price", 0) for item in data_houses if item.get("price") is not None),
    "rooms": max(
        int(str(item.get("rooms", "0 rooms")).split()[0])
        for item in data_houses
        if item.get("rooms") and str(item.get("rooms")).split()
    ),
    "storage_rating": max(
        item.get("storage_rating", 0) for item in data_houses if item.get("storage_rating") is not None),
    "sun_factor": max(item.get("sun_factor", 0) for item in data_houses if item.get("sun_factor") is not None),
    "crime_rating": max(district.get("crime_rating", 0) for district in data_districts),
    "public_transport_rating": max(district.get("public_transport_rating", 0) for district in data_districts),
    "school_rating": max(school.get("rating", 0) for school in data_schools),
    "school_capacity": max(school.get("capacity", 0) for school in data_schools),
    "school_built_year": max(school.get("built_year", 0) for school in data_schools)
}

all_fields = [
    "advertisement", "agent_id", "bathrooms", "condition_rating",
    "days_on_marked", "external_storage_m2", "fireplace", "kitchens",
    "lot_w", "parking", "years_since_remodeled", "rooms", "size", "sold",
    "sold_in_month", "storage_rating", "sun_factor", "year", "house_age",
    "crime_rating", "public_transport_rating", "school_rating", "capacity", "built_year"
    # Note: 'color' is removed from 'all_fields' as we are using one-hot encoding for it
]

categorical_fields = ["advertisement", "agent_id", "fireplace", "parking", "sold", "sold_in_month"]
# 'color' is handled separately for one-hot encoding

numeric_fields = [
    "bathrooms", "condition_rating", "days_on_marked", "external_storage_m2",
    "kitchens", "lot_w", "parking", "years_since_remodeled", "rooms", "size", "storage_rating",
    "sun_factor", "year", "house_age", "crime_rating", "public_transport_rating",
    "school_rating", "capacity", "built_year"
]

def impute_missing_values(houses):
    for feature in numeric_fields:
        values = [h[feature] for h in houses if feature in h and isinstance(h[feature], (int, float))]
        if values:
            mean_val = np.mean(values)
            for h in houses:
                if feature not in h or not isinstance(h[feature], (int, float)):
                    h[feature] = mean_val
        else:
            for h in houses:
                h[feature] = 0
    for feature in categorical_fields:
        for h in houses:
            val = h.get(feature, None)
            if val is None or not isinstance(val, str) or val.strip() == "":
                h[feature] = "unknown"

def encode_rooms(h):
    r = h.get("rooms", "0 rooms")
    try:
        num_r = int(str(r).split()[0])
    except:
        num_r = 0
    h["rooms"] = num_r

def add_house_age(houses, current_year=2023):
    for h in houses:
        if 'year' in h and isinstance(h['year'], (int, float)):
            h['house_age'] = current_year - h['year']
        else:
            h['house_age'] = 0

def encode_categorical_fields(houses):
    cat_maps = {}
    for f in categorical_fields:
        cats = set(h[f] for h in houses)
        cat_maps[f] = {c: i for i, c in enumerate(sorted(cats))}
    return cat_maps

def apply_categorical_encoding(h, cat_maps):
    for f in categorical_fields:
        val = h.get(f, "unknown")
        if val not in cat_maps[f]:
            if "unknown" in cat_maps[f]:
                h[f] = cat_maps[f]["unknown"]
            else:
                h[f] = 0
        else:
            h[f] = cat_maps[f][val]

def one_hot_encode_field(houses, field):
    unique_values = sorted(set(h.get(field, "unknown") for h in houses))
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}

    for h in houses:
        one_hot_vector = [0] * len(unique_values)
        value = h.get(field, "unknown")
        if value in value_to_index:
            one_hot_vector[value_to_index[value]] = 1
        else:
            # If value not found, default to 'unknown' or zeros
            if "unknown" in value_to_index:
                one_hot_vector[value_to_index["unknown"]] = 1
        h[f"{field}_onehot"] = one_hot_vector
        print(f"one_hot_func= {unique_values}, {one_hot_vector}")
    return unique_values

def prepare_dataset(houses, cat_maps, feature_order, advertisement_categories, agent_id_categories, color_categories):
    X = []
    y = []
    for h in houses:
        apply_categorical_encoding(h, cat_maps)
        row = []
        # Include one-hot encoded 'advertisement' field
        for adv_value in advertisement_categories:
            idx = advertisement_categories.index(adv_value)
            row.append(h['advertisement_onehot'][idx])

        # Include one-hot encoded 'agent_id' field
        for agent_value in agent_id_categories:
            idx = agent_id_categories.index(agent_value)
            row.append(h['agent_id_onehot'][idx])

        # Include one-hot encoded 'color' field
        for color_value in color_categories:
            idx = color_categories.index(color_value)
            row.append(h['color_onehot'][idx])

        # Include other numeric features based on 'feature_order'
        for f in feature_order[len(advertisement_categories) + len(agent_id_categories) + len(color_categories):]:
            row.append(float(h[f]))
        X.append(row)
        y.append(h['price'])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def train_test_split_custom(X, y, test_size=0.2):
    data_size = len(X)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    test_set_size = int(data_size * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

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
    dw = (1 / m) * np.dot(X.T, residuals)
    db = (1 / m) * np.sum(residuals)
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
                print(f"Training stopped due to NaN/Inf loss at epoch {epoch}, batch {start_idx}.")
                save_state_to_file(best_w, best_b, best_loss, state_file)
                return best_w, best_b, best_loss

            dw, db = compute_gradients_batch(X_batch, y_batch, y_pred_batch)
            if np.isnan(dw).any() or np.isnan(db) or np.isinf(dw).any() or np.isinf(db):
                print(f"Invalid gradients at epoch {epoch}, batch {start_idx}.")
                save_state_to_file(best_w, best_b, best_loss, state_file)
                return best_w, best_b, best_loss

            best_w, best_b = update_parameters(best_w, best_b, dw, db, learning_rate)

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

def save_model(w, b, X_train_means, X_train_stds, y_train_mean, y_train_std, feature_order, cat_maps,
               advertisement_categories, agent_id_categories, color_categories, filename='linear_regression_model.pkl'):
    model = {
        'weights': w,
        'bias': b,
        'X_train_means': X_train_means,
        'X_train_stds': X_train_stds,
        'y_train_mean': y_train_mean,
        'y_train_std': y_train_std,
        'feature_order': feature_order,
        'cat_maps': cat_maps,
        'advertisement_categories': advertisement_categories,
        'agent_id_categories': agent_id_categories,
        'color_categories': color_categories
    }
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename='linear_regression_model.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return (model['weights'], model['bias'], model['X_train_means'], model['X_train_stds'],
            model['y_train_mean'], model['y_train_std'], model['feature_order'], model['cat_maps'],
            model['advertisement_categories'], model['agent_id_categories'], model['color_categories'])

def preprocess_user_input(house_data, feature_order, X_train_means, X_train_stds, cat_maps, advertisement_categories,
                          agent_id_categories, color_categories):
    # Process 'house_age'
    if 'year' in house_data and house_data['year'] is not None:
        try:
            y = float(house_data['year'])
            house_data['house_age'] = 2023 - y
        except:
            house_data['house_age'] = 0
    else:
        house_data['house_age'] = 0

    # Process 'rooms'
    if 'rooms' in house_data:
        r_str = str(house_data['rooms'])
        try:
            house_data['rooms'] = float(r_str.split()[0])
        except:
            house_data['rooms'] = 0
    else:
        house_data['rooms'] = 0

    # One-hot encode 'advertisement' field
    advertisement_onehot = [0] * len(advertisement_categories)
    advertisement_value = house_data.get('advertisement', 'unknown')
    if advertisement_value not in advertisement_categories:
        advertisement_value = 'unknown'
    if advertisement_value in advertisement_categories:
        idx = advertisement_categories.index(advertisement_value)
        advertisement_onehot[idx] = 1
    house_data['advertisement_onehot'] = advertisement_onehot

    # One-hot encode 'agent_id' field
    agent_onehot = [0] * len(agent_id_categories)
    agent_value = house_data.get('agent_id', 'unknown')
    if agent_value not in agent_id_categories:
        agent_value = 'unknown'
    if agent_value in agent_id_categories:
        idx = agent_id_categories.index(agent_value)
        agent_onehot[idx] = 1
    house_data['agent_id_onehot'] = agent_onehot

    # One-hot encode 'color' field
    color_onehot = [0] * len(color_categories)
    color_value = house_data.get('color', 'unknown')
    if color_value not in color_categories:
        color_value = 'unknown'
    if color_value in color_categories:
        idx = color_categories.index(color_value)
        color_onehot[idx] = 1
    house_data['color_onehot'] = color_onehot

    # Map 'district_id' to 'crime_rating' and 'public_transport_rating'
    district_id = house_data.get('district_id', 'unknown')
    if district_id in districts_dict:
        house_data['crime_rating'] = districts_dict[district_id].get('crime_rating', 0)
        house_data['public_transport_rating'] = districts_dict[district_id].get('public_transport_rating', 0)
    else:
        house_data['crime_rating'] = 0
        house_data['public_transport_rating'] = 0

    # Map 'school_id' to 'school_rating', 'capacity', and 'built_year'
    school_id = house_data.get('school_id', 'unknown')
    if school_id in schools_dict:
        house_data['school_rating'] = schools_dict[school_id].get('rating', 0)
        house_data['capacity'] = schools_dict[school_id].get('capacity', 0)
        house_data['built_year'] = schools_dict[school_id].get('built_year', 0)
    else:
        house_data['school_rating'] = 0
        house_data['capacity'] = 0
        house_data['built_year'] = 0

    # Process 'remodeled' field
    if 'remodeled' in house_data:
        if house_data['remodeled'] == -1:
            house_data['years_since_remodeled'] = 0  # Indicates never remodeled
        else:
            try:
                remodeled_year = float(house_data['remodeled'])
                house_data['years_since_remodeled'] = 2023 - remodeled_year
                if house_data['years_since_remodeled'] < 0:
                    house_data['years_since_remodeled'] = 0  # Handle future years gracefully
            except:
                house_data['years_since_remodeled'] = 0
    else:
        house_data['years_since_remodeled'] = 0

    # Remove or rename 'remodeled' if necessary
    # For this implementation, we'll remove 'remodeled' as we have 'years_since_remodeled'
    if 'remodeled' in house_data:
        del house_data['remodeled']

    # Process other categorical fields
    for f in categorical_fields:
        if f in ['advertisement', 'agent_id']:
            continue  # Already handled
        val = house_data.get(f, "unknown")
        if not val or not isinstance(val, str):
            val = "unknown"
        if val not in cat_maps[f]:
            if "unknown" in cat_maps[f]:
                val = "unknown"
            else:
                val = list(cat_maps[f].keys())[0]
        house_data[f] = cat_maps[f][val]

    # Process numeric fields
    for f in numeric_fields:
        if f in ['advertisement', 'agent_id', 'color', 'crime_rating', 'public_transport_rating', 'school_rating', 'capacity', 'built_year']:
            continue  # Already handled or added
        val = house_data.get(f, None)
        if val is None:
            house_data[f] = 0
        else:
            try:
                house_data[f] = float(val)
            except:
                house_data[f] = 0

    # Build processed_data according to feature_order
    processed_data = []
    for f in feature_order:
        if f.startswith('advertisement_'):
            adv_val = f[len('advertisement_'):]
            idx = advertisement_categories.index(adv_val)
            processed_data.append(house_data['advertisement_onehot'][idx])
        elif f.startswith('agent_id_'):
            agent_val = f[len('agent_id_'):]
            idx = agent_id_categories.index(agent_val)
            processed_data.append(house_data['agent_id_onehot'][idx])
        elif f.startswith('color_'):
            color_val = f[len('color_'):]
            idx = color_categories.index(color_val)
            processed_data.append(house_data['color_onehot'][idx])
        elif f == 'years_since_remodeled':
            processed_data.append(float(house_data.get('years_since_remodeled', 0)))
        elif f in ['crime_rating', 'public_transport_rating', 'school_rating', 'capacity', 'built_year']:
            processed_data.append(float(house_data.get(f, 0)))
        else:
            processed_data.append(float(house_data.get(f, 0)))

    processed_data = np.array(processed_data)
    processed_data_scaled = (processed_data - X_train_means) / X_train_stds
    return processed_data_scaled

@app.route('/')
def index():
    # Prepare dropdown lists
    agents = sorted(data_agents, key=lambda x: x.get('agent_id', 'unknown'))
    agent_options = [{'id': a['agent_id'], 'name': a.get('name', 'unknown')} for a in agents]

    district_ids_list = sorted({d['id'] for d in data_districts if 'id' in d})
    school_ids_list = sorted({s['id'] for s in data_schools if 'id' in s})

    # If no known category, we can rely on 'unknown'
    colors_list = sorted(unique_values["colors"])
    advertisement_list = unique_values["advertisement_types"]

    # Define months
    months_list = ["unknown", "January", "February", "March", "April", "May", "June", "July", "August", "September",
                   "October", "November", "December"]

    return render_template('index.html',
                           unique_values=unique_values,
                           maximums=maximums,
                           agent_options=agent_options,
                           district_ids_list=district_ids_list,
                           school_ids_list=school_ids_list,
                           colors_list=colors_list,
                           advertisement_list=advertisement_list,
                           months_list=months_list
                           )

@app.route('/predict', methods=['POST'])
def predict_route():
    user_input = request.get_json()
    w, b, X_train_means, X_train_stds, y_train_mean, y_train_std, feature_order, cat_maps, advertisement_categories, agent_id_categories, color_categories = load_model()

    features_scaled = preprocess_user_input(user_input, feature_order, X_train_means, X_train_stds, cat_maps,
                                            advertisement_categories, agent_id_categories, color_categories)
    predicted_price_scaled = predict(features_scaled.reshape(1, -1), w, b)
    predicted_price = (predicted_price_scaled * y_train_std) + y_train_mean
    rounded_price = round(predicted_price.item(), 2)
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
        # Load data
        agents = load_jsonl('data/agents.jsonl')
        districts = load_jsonl('data/districts.jsonl')
        schools = load_jsonl('data/schools.jsonl')
        houses = load_jsonl('data/houses.jsonl')

        # Fill missing fields with 'unknown'
        for h in houses:
            if 'rooms' not in h:
                h['rooms'] = "0 rooms"
            if 'advertisement' not in h:
                h['advertisement'] = "unknown"
            if 'fireplace' not in h:
                h['fireplace'] = "unknown"
            if 'parking' not in h:
                h['parking'] = "unknown"
            if 'sold' not in h:
                h['sold'] = "unknown"
            if 'sold_in_month' not in h:
                h['sold_in_month'] = "unknown"
            if 'color' not in h:
                h['color'] = "unknown"
            if 'agent_id' not in h:
                h['agent_id'] = "unknown"
            if 'district_id' not in h:
                h['district_id'] = "unknown"
            if 'school_id' not in h:
                h['school_id'] = "unknown"
            if 'remodeled' not in h:
                h['remodeled'] = -1  # Assume -1 if missing

        # Populate additional features from districts and schools
        for house in houses:
            # Add district features
            district_id = house.get('district_id', 'unknown')
            if district_id in districts_dict:
                house['crime_rating'] = districts_dict[district_id].get('crime_rating', 0)
                house['public_transport_rating'] = districts_dict[district_id].get('public_transport_rating', 0)
            else:
                house['crime_rating'] = 0
                house['public_transport_rating'] = 0

            # Add school features
            school_id = house.get('school_id', 'unknown')
            if school_id in schools_dict:
                house['school_rating'] = schools_dict[school_id].get('rating', 0)
                house['capacity'] = schools_dict[school_id].get('capacity', 0)
                house['built_year'] = schools_dict[school_id].get('built_year', 0)
            else:
                house['school_rating'] = 0
                house['capacity'] = 0
                house['built_year'] = 0

        # Encode 'rooms' and add 'house_age'
        for h in houses:
            encode_rooms(h)

        add_house_age(houses, current_year=2023)

        # Handle 'remodeled' field: Replace -1 with 0 (never remodeled), else compute years since remodeled
        for h in houses:
            if h['remodeled'] == -1:
                h['years_since_remodeled'] = 0  # Indicates never remodeled
            else:
                try:
                    remodeled_year = float(h['remodeled'])
                    h['years_since_remodeled'] = 2023 - remodeled_year
                    if h['years_since_remodeled'] < 0:
                        h['years_since_remodeled'] = 0  # Handle future years gracefully
                except:
                    h['years_since_remodeled'] = 0
            # Remove the original 'remodeled' field as we have 'years_since_remodeled'
            del h['remodeled']

        impute_missing_values(houses)

        # One-hot encode the "advertisement" field
        advertisement_categories = one_hot_encode_field(houses, "advertisement")
        print("One-hot encoded categories for 'advertisement':", advertisement_categories)

        # One-hot encode the "agent_id" field
        agent_id_categories = one_hot_encode_field(houses, "agent_id")
        print("One-hot encoded categories for 'agent_id':", agent_id_categories)

        # One-hot encode the "color" field
        color_categories = one_hot_encode_field(houses, "color")
        print("One-hot encoded categories for 'color':", color_categories)

        # Encode categorical fields
        cat_maps = encode_categorical_fields(houses)

        # Create feature order including one-hot encoded 'advertisement', 'agent_id', and 'color' fields
        one_hot_advertisement_fields = [f"advertisement_{val}" for val in advertisement_categories]
        one_hot_agent_fields = [f"agent_id_{val}" for val in agent_id_categories]
        one_hot_color_fields = [f"color_{val}" for val in color_categories]
        feature_order = one_hot_advertisement_fields + one_hot_agent_fields + one_hot_color_fields + [
            f for f in all_fields if f not in ['advertisement', 'agent_id', 'color']
        ]

        # Prepare dataset
        X, y = prepare_dataset(houses, cat_maps, feature_order, advertisement_categories, agent_id_categories,
                               color_categories)

        # Check for NaN or infinite values
        if np.isnan(X).any() or np.isinf(X).any():
            print("NaN or infinite values detected in feature matrix X.")
        if np.isnan(y).any() or np.isinf(y).any():
            print("NaN or infinite values detected in target vector y.")

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split_custom(X, y)

        # Feature scaling
        X_train_means = np.mean(X_train, axis=0)
        X_train_stds = np.std(X_train, axis=0)
        X_train_stds[X_train_stds == 0] = 1  # To avoid division by zero
        X_train_scaled = (X_train - X_train_means) / X_train_stds
        X_test_scaled = (X_test - X_train_means) / X_train_stds

        # Target scaling
        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train)
        if y_train_std == 0:
            y_train_std = 1
        y_train_scaled = (y_train - y_train_mean) / y_train_std
        y_test_scaled = (y_test - y_train_mean) / y_train_std

        # Train model
        w, b, best_loss = train_linear_regression(X_train_scaled, y_train_scaled, learning_rate=0.001, epochs=2000,
                                                  batch_size=32)
        print(f'Best Loss: {best_loss}')
        print('Learned Weights:', w)
        print('Bias:', b)

        # Evaluate model
        evaluate_model(X_test_scaled, y_test_scaled, w, b)

        # Save model
        save_model(w, b, X_train_means, X_train_stds, y_train_mean, y_train_std, feature_order, cat_maps,
                   advertisement_categories, agent_id_categories, color_categories)

    app.run(debug=True)
