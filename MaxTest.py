import json
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


data_houses = load_jsonl("data/houses.jsonl")
data_districts = load_jsonl("data/districts.jsonl")
data_agents = load_jsonl("data/agents.jsonl")
data_schools = load_jsonl("data/schools.jsonl")

districts = {item["id"] for item in data_districts}

unique_values = {
    "colors": list(set(item.get("color", "unknown") for item in data_houses)),
    "district_ids": sorted(list(districts)),
    "ratings": list(range(6)),
    "condition_ratings": list(range(11)),
}

maximums = {
    "year": max(
        item.get("year", 0) for item in data_houses if item.get("year") is not None
    ),
    "remodeled": max(
        item.get("remodeled", 0)
        for item in data_houses
        if item.get("remodeled") is not None
    ),
    "bathrooms": max(
        item.get("bathrooms", 0)
        for item in data_houses
        if item.get("bathrooms") is not None
    ),
    "days_on_marked": max(
        item.get("days_on_marked", 0)
        for item in data_houses
        if item.get("days_on_marked") is not None
    ),
    "external_storage_m2": max(
        item.get("external_storage_m2", 0)
        for item in data_houses
        if item.get("external_storage_m2") is not None
    ),
    "kitchens": max(
        item.get("kitchens", 0)
        for item in data_houses
        if item.get("kitchens") is not None
    ),
    "lot_w": max(
        item.get("lot_w", 0) for item in data_houses if item.get("lot_w") is not None
    ),
    "price": max(
        item.get("price", 0) for item in data_houses if item.get("price") is not None
    ),
    "rooms": max(
        int(str(item.get("rooms", "0 rooms")).split()[0])
        for item in data_houses
        if item.get("rooms") and str(item.get("rooms")).split()
    ),
    "storage_rating": max(
        item.get("storage_rating", 0)
        for item in data_houses
        if item.get("storage_rating") is not None
    ),
    "sun_factor": max(
        item.get("sun_factor", 0)
        for item in data_houses
        if item.get("sun_factor") is not None
    ),
}


all_fields = [
    "advertisement",
    "agent_id",
    "bathrooms",
    "condition_rating",
    "days_on_marked",
    "district_id",
    "external_storage_m2",
    "fireplace",
    "kitchens",
    "lot_w",
    "parking",
    "remodeled",
    "rooms",
    "school_id",
    "size",
    "sold",
    "sold_in_month",
    "storage_rating",
    "sun_factor",
    "year",
    "house_age",
    # Note: 'color' is removed from 'all_fields' as we are using one-hot encoding for it
]

categorical_fields = [
    "advertisement",
    "agent_id",
    "district_id",
    "fireplace",
    "parking",
    "school_id",
    "sold",
    "sold_in_month",
]
# 'color' is handled separately for one-hot encoding

numeric_fields = [f for f in all_fields if f not in categorical_fields]


def impute_missing_values(houses):
    for feature in numeric_fields:
        values = [
            h[feature]
            for h in houses
            if feature in h and isinstance(h[feature], (int, float))
        ]
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


def encode_schools(houses, schools):
    school_scores = {}
    for school in schools:
        score = (
            school.get("rating", 0)  # Higher rating is better
            +
            # Normalize capacity to a smaller scale
            school.get("capacity", 0) / 100
        )
        school_scores[school["id"]] = score

    min_score = min(school_scores.values())
    max_score = max(school_scores.values())

    # Normalize scores to [0, 1]
    for school_id in school_scores:
        school_scores[school_id] = (school_scores[school_id] - min_score) / (
            max_score - min_score
        )

    for house in houses:
        house["school_score"] = school_scores.get(house["school_id"], 0)


def encode_districts(houses, districts):
    district_scores = {}
    for district in districts:
        score = (
            district.get("crime_rating", 0) * -1  # Lower crime is better
            +
            # Higher transport rating is better
            district.get("public_transport_rating", 0)
        )
        district_scores[district["id"]] = score

    min_score = min(district_scores.values())
    max_score = max(district_scores.values())

    # Normalize scores to [0, 1]
    for district_id in district_scores:
        district_scores[district_id] = (district_scores[district_id] - min_score) / (
            max_score - min_score
        )

    for house in houses:
        house["district_score"] = district_scores.get(house["district_id"], 0)


def encode_rooms(h):
    r = h.get("rooms", "0 rooms")
    try:
        num_r = int(str(r).split()[0])
    except:
        num_r = 0
    h["rooms"] = num_r


def add_house_age(houses, current_year=2023):
    for h in houses:
        if "year" in h and isinstance(h["year"], (int, float)):
            h["house_age"] = current_year - h["year"]
        else:
            h["house_age"] = 0


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
        print("one_hot_func= ", unique_values, one_hot_vector)
    return unique_values


def prepare_dataset(houses, cat_maps, feature_order, color_categories):
    X = []
    y = []
    for h in houses:
        apply_categorical_encoding(h, cat_maps)
        row = []
        # Include one-hot encoded 'color' field
        for color_value in color_categories:
            idx = color_categories.index(color_value)
            row.append(h["color_onehot"][idx])

        # Include other features based on 'feature_order'
        # Skip the one-hot color fields
        for f in feature_order[len(color_categories) :]:
            row.append(float(h[f]))
        X.append(row)
        y.append(h["price"])
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
    loss = (1 / (2 * m)) * np.sum(residuals**2)
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
    state = {"weights": best_w.tolist(), "bias": best_b, "best_loss": best_loss}
    with open(file_path, "w") as f:
        json.dump(state, f)


def load_state_from_file(file_path):
    with open(file_path, "r") as f:
        state = json.load(f)
    best_w = np.array(state["weights"])
    best_b = state["bias"]
    best_loss = state["best_loss"]
    print(f"State loaded from {file_path}")
    return best_w, best_b, best_loss


def train_linear_regression(
    X_train,
    y_train_scaled,
    learning_rate=0.001,
    epochs=2000,
    batch_size=32,
    state_file="model_state.json",
):
    n_samples, n_features = X_train.shape

    if os.path.exists(state_file):
        os.remove(state_file)
    if os.path.exists("linear_regression_model.pkl"):
        os.remove("linear_regression_model.pkl")

    best_w, best_b = initialize_parameters(n_features)
    best_loss = float("inf")

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
                print(
                    f"Training stopped due to NaN/Inf loss at epoch {epoch}, batch {start_idx}."
                )
                save_state_to_file(best_w, best_b, best_loss, state_file)
                return best_w, best_b, best_loss

            dw, db = compute_gradients_batch(X_batch, y_batch, y_pred_batch)
            if np.isnan(dw).any() or np.isnan(db) or np.isinf(dw).any() or np.isinf(db):
                print(
                    f"Invalid gradients at epoch {
                      epoch}, batch {start_idx}."
                )
                save_state_to_file(best_w, best_b, best_loss, state_file)
                return best_w, best_b, best_loss

            best_w, best_b = update_parameters(best_w, best_b, dw, db, learning_rate)

        y_pred = predict(X_train, best_w, best_b)
        total_loss = compute_loss(y_train_scaled, y_pred)

        if total_loss < best_loss:
            best_loss = total_loss
            save_state_to_file(best_w, best_b, best_loss, state_file)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}, Best Loss: {best_loss}")

    return best_w, best_b, best_loss


def evaluate_model(X_test, y_test_scaled, w, b):
    y_pred_scaled = predict(X_test, w, b)
    mse = compute_loss(y_test_scaled, y_pred_scaled) * 2
    rmse = np.sqrt(mse)
    ss_total = np.sum((y_test_scaled - np.mean(y_test_scaled)) ** 2)
    ss_residual = np.sum((y_test_scaled - y_pred_scaled) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    print(f"Test MSE: {mse}")
    print(f"Test RMSE: {rmse}")
    print(f"R-squared: {r_squared}")


def save_model(
    w,
    b,
    X_train_means,
    X_train_stds,
    y_train_mean,
    y_train_std,
    feature_order,
    cat_maps,
    color_categories,
    filename="linear_regression_model.pkl",
):
    model = {
        "weights": w,
        "bias": b,
        "X_train_means": X_train_means,
        "X_train_stds": X_train_stds,
        "y_train_mean": y_train_mean,
        "y_train_std": y_train_std,
        "feature_order": feature_order,
        "cat_maps": cat_maps,
        "color_categories": color_categories,
    }
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


def load_model(filename="linear_regression_model.pkl"):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return (
        model["weights"],
        model["bias"],
        model["X_train_means"],
        model["X_train_stds"],
        model["y_train_mean"],
        model["y_train_std"],
        model["feature_order"],
        model["cat_maps"],
        model["color_categories"],
    )


def preprocess_user_input(
    house_data, feature_order, X_train_means, X_train_stds, cat_maps, color_categories
):
    # Process 'house_age'
    if "year" in house_data and house_data["year"] is not None:
        try:
            y = float(house_data["year"])
            house_data["house_age"] = 2023 - y
        except:
            house_data["house_age"] = 0
    else:
        house_data["house_age"] = 0

    # Process 'rooms'
    if "rooms" in house_data:
        r_str = str(house_data["rooms"])
        try:
            house_data["rooms"] = float(r_str.split()[0])
        except:
            house_data["rooms"] = 0
    else:
        house_data["rooms"] = 0

    # One-hot encode 'color' field
    color_onehot = [0] * len(color_categories)
    color_value = house_data.get("color", "unknown")
    if color_value not in color_categories:
        color_value = "unknown"
    idx = color_categories.index(color_value)
    color_onehot[idx] = 1
    house_data["color_onehot"] = color_onehot

    # Process other categorical fields
    for f in categorical_fields:
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
        if f.startswith("color_"):
            color_value = f[len("color_") :]
            idx = color_categories.index(color_value)
            processed_data.append(house_data["color_onehot"][idx])
        else:
            processed_data.append(float(house_data.get(f, 0)))

    processed_data = np.array(processed_data)
    processed_data_scaled = (processed_data - X_train_means) / X_train_stds
    return processed_data_scaled


@app.route("/")
def index():
    # Prepare dropdown lists
    agent_ids = sorted({a["agent_id"] for a in data_agents if "agent_id" in a})
    district_ids_list = sorted({d["id"] for d in data_districts if "id" in d})
    school_ids_list = sorted({s["id"] for s in data_schools if "id" in s})

    # If no known category, we can rely on 'unknown'
    colors_list = sorted(unique_values["colors"])

    # Define months
    months_list = [
        "unknown",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    return render_template(
        "index.html",
        unique_values=unique_values,
        maximums=maximums,
        agent_ids=agent_ids,
        district_ids_list=district_ids_list,
        school_ids_list=school_ids_list,
        colors_list=colors_list,
        months_list=months_list,
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    user_input = request.get_json()
    (
        w,
        b,
        X_train_means,
        X_train_stds,
        y_train_mean,
        y_train_std,
        feature_order,
        cat_maps,
        color_categories,
    ) = load_model()

    features_scaled = preprocess_user_input(
        user_input,
        feature_order,
        X_train_means,
        X_train_stds,
        cat_maps,
        color_categories,
    )
    predicted_price_scaled = predict(features_scaled.reshape(1, -1), w, b)
    predicted_price = (predicted_price_scaled * y_train_std) + y_train_mean
    rounded_price = round(predicted_price.item(), 2)
    return jsonify({"price": rounded_price}), 200


@app.route("/about")
def about():
    return "This is a test Flask app for DashEstate."


@app.route("/routes")
def list_routes():
    route_list = []
    for rule in app.url_map.iter_rules():
        route_info = {
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "rule": rule.rule,
        }
        route_list.append(route_info)
    return jsonify(route_list), 200


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        agents = load_jsonl("data/agents.jsonl")
        districts = load_jsonl("data/districts.jsonl")
        schools = load_jsonl("data/schools.jsonl")
        houses = load_jsonl("data/houses.jsonl")

        for h in houses:
            if "rooms" not in h:
                h["rooms"] = "0 rooms"
            if "advertisement" not in h:
                h["advertisement"] = "unknown"
            if "fireplace" not in h:
                h["fireplace"] = "unknown"
            if "parking" not in h:
                h["parking"] = "unknown"
            if "sold" not in h:
                h["sold"] = "unknown"
            if "sold_in_month" not in h:
                h["sold_in_month"] = "unknown"
            if "color" not in h:
                h["color"] = "unknown"
            if "agent_id" not in h:
                h["agent_id"] = "unknown"
            if "district_id" not in h:
                h["district_id"] = "unknown"
            if "school_id" not in h:
                h["school_id"] = "unknown"

        agents_dict = {agent["agent_id"]: agent for agent in agents}
        districts_dict = {district["id"]: district for district in districts}
        schools_dict = {school["id"]: school for school in schools}

        for house in houses:
            agent_info = agents_dict.get(house.get("agent_id"), {})
            house["agent_name"] = agent_info.get("name", "Unknown")
            district_info = districts_dict.get(house.get("district_id"), {})
            house["crime_rating"] = district_info.get("crime_rating", None)
            house["public_transport_rating"] = district_info.get(
                "public_transport_rating", None
            )
            school_info = schools_dict.get(house.get("school_id"), {})
            house["school_rating"] = school_info.get("rating", None)
            house["school_capacity"] = school_info.get("capacity", None)
            house["school_built_year"] = school_info.get("built_year", None)

        for h in houses:
            encode_rooms(h)

        add_house_age(houses, current_year=2023)
        impute_missing_values(houses)

        # One-hot encode the "color" field
        color_categories = one_hot_encode_field(houses, "color")
        print("One-hot encoded categories for 'color':", color_categories)

        cat_maps = encode_categorical_fields(houses)

        # Create feature order including one-hot encoded 'color' fields
        one_hot_color_fields = [f"color_{val}" for val in color_categories]
        feature_order = one_hot_color_fields + [f for f in all_fields if f != "color"]

        X, y = prepare_dataset(houses, cat_maps, feature_order, color_categories)

        if np.isnan(X).any() or np.isinf(X).any():
            print("NaN or infinite values detected in feature matrix X.")
        if np.isnan(y).any() or np.isinf(y).any():
            print("NaN or infinite values detected in target vector y.")

        X_train, X_test, y_train, y_test = train_test_split_custom(X, y)

        X_train_means = np.mean(X_train, axis=0)
        X_train_stds = np.std(X_train, axis=0)
        X_train_stds[X_train_stds == 0] = 1
        X_train_scaled = (X_train - X_train_means) / X_train_stds
        X_test_scaled = (X_test - X_train_means) / X_train_stds

        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train)
        if y_train_std == 0:
            y_train_std = 1
        y_train_scaled = (y_train - y_train_mean) / y_train_std
        y_test_scaled = (y_test - y_train_mean) / y_train_std

        w, b, best_loss = train_linear_regression(
            X_train_scaled,
            y_train_scaled,
            learning_rate=0.001,
            epochs=2000,
            batch_size=32,
        )
        print(f"Best Loss: {best_loss}")
        print("Learned Weights:", w)
        print("Bias:", b)

        evaluate_model(X_test_scaled, y_test_scaled, w, b)

        save_model(
            w,
            b,
            X_train_means,
            X_train_stds,
            y_train_mean,
            y_train_std,
            feature_order,
            cat_maps,
            color_categories,
        )

    app.run(debug=True)
