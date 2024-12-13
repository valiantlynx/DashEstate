import json
import requests
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

# Load the houses data


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            print(line)
            data.append(json.loads(line))
    return data


# Test the API and evaluate the predictions


def test_predictions(api_url, houses_data):
    y_true = []
    y_pred = []

    for house in houses_data:
        # Make a POST request to the API
        response = requests.post(f"{api_url}/predict", json=house)
        if response.status_code == 200:
            result = response.json()
            if "price" in result:
                y_pred.append(result["price"])
                y_true.append(house.get("price", 0))
            else:
                print(
                    f"Error in prediction for house: {
                        house}, Response: {result}"
                )
        else:
            print(
                f"Failed to get prediction. Status Code: {
                    response.status_code}, Response: {response.text}"
            )

    return np.array(y_true), np.array(y_pred)


# Calculate and display metrics


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    avg_absolute_diff = np.mean(np.abs(y_true - y_pred))
    avg_percentage_diff = np.mean(np.abs(y_true - y_pred) / y_true) * 100

    print("Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")
    print(f"Average Absolute Difference: {avg_absolute_diff:.2f}")
    print(f"Average Percentage Difference: {avg_percentage_diff:.2f}%")

    return mse, r2, avg_absolute_diff, avg_percentage_diff


# Plot the Martin plot using Plotly


def plot_martin_plotly(y_true, y_pred):
    # Ideal line
    ideal_line = [min(y_true), max(y_true)]

    # Create the figure
    fig = go.Figure()

    # Add scatter plot for predictions
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Predictions",
            marker=dict(color="blue", size=6, opacity=0.7),
        )
    )

    # Add ideal line
    fig.add_trace(
        go.Scatter(
            x=ideal_line,
            y=ideal_line,
            mode="lines",
            name="Ideal",
            line=dict(color="red", dash="dash"),
        )
    )

    # Customize layout
    fig.update_layout(
        title="Martin Plot (Actual vs Predicted Prices)",
        xaxis_title="Actual Price",
        yaxis_title="Predicted Price",
        legend=dict(x=0.1, y=1.1, orientation="h"),
        width=800,
        height=800,
        template="plotly_white",
    )

    # Show the plot
    fig.show()


if __name__ == "__main__":
    # Backend API URL
    API_URL = "http://localhost:5000"

    # Load data
    houses = load_jsonl("data/houses.jsonl")

    # Test predictions
    y_true, y_pred = test_predictions(API_URL, houses)

    # Calculate metrics and display
    if len(y_true) > 0 and len(y_pred) > 0:
        mse, r2, avg_abs_diff, avg_pct_diff = calculate_metrics(y_true, y_pred)
        # Plot the results using Plotly
        plot_martin_plotly(y_true, y_pred)
    else:
        print("No predictions could be made.")
