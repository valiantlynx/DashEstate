import json
import requests
import numpy as np
import plotly.graph_objects as go


# Load the houses data
def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Filter unsold houses
def filter_unsold_houses(houses_data):
    return [house for house in houses_data if house.get("sold", "no") == "no"]


# Test the API and evaluate the predictions
def test_predictions(api_url, houses_data):
    y_true = []
    y_pred = []
    savings = []

    for house in houses_data:
        # Make a POST request to the API
        response = requests.post(f"{api_url}/predict", json=house)
        if response.status_code == 200:
            result = response.json()
            if "price" in result:
                predicted_price = result["price"]
                actual_price = house.get("price", 0)

                y_pred.append(predicted_price)
                y_true.append(actual_price)

                # Calculate potential savings
                if predicted_price > actual_price:
                    savings.append(predicted_price - actual_price)
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

    house_number = len(savings) if savings else 0
    sum_savings = np.sum(savings) if savings else 0
    return np.array(y_true), np.array(y_pred), sum_savings, house_number


# Plot houses worth bidding on vs not worth it
def plot_worth_bidding(houses_data, y_true, y_pred, sum_savings, house_number):
    worth_bidding = []
    not_worth_bidding = []
    savings_per_house = []

    for true_price, pred_price in zip(y_true, y_pred):
        savings = max(0, pred_price - true_price)
        savings_per_house.append(savings)
        if pred_price > true_price:
            worth_bidding.append((true_price, pred_price, savings))
        else:
            not_worth_bidding.append((true_price, pred_price, savings))

    fig = go.Figure()

    # Plot worth bidding houses
    fig.add_trace(
        go.Scatter(
            x=[pair[0] for pair in worth_bidding],
            y=[pair[1] for pair in worth_bidding],
            mode="markers",
            name="Worth Bidding",
            marker=dict(color="green", size=8, opacity=0.7),
            text=[f"Savings: {pair[2]:.2f} NOK" for pair in worth_bidding],
            hoverinfo="text+x+y",
        )
    )

    # Plot not worth bidding houses
    fig.add_trace(
        go.Scatter(
            x=[pair[0] for pair in not_worth_bidding],
            y=[pair[1] for pair in not_worth_bidding],
            mode="markers",
            name="Not Worth Bidding",
            marker=dict(color="red", size=8, opacity=0.7),
            text=[f"Savings: {pair[2]:.2f} NOK" for pair in not_worth_bidding],
            hoverinfo="text+x+y",
        )
    )

    # Add ideal line
    ideal_line = [min(y_true), max(y_true)]
    fig.add_trace(
        go.Scatter(
            x=ideal_line,
            y=ideal_line,
            mode="lines",
            name="Ideal",
            line=dict(color="blue", dash="dash"),
        )
    )

    # Annotate with sum savings and house count
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        text=f"<b>Total Savings: {sum_savings:,.2f} NOK on {
            house_number} houses</b>",
        showarrow=False,
        font=dict(size=16, color="purple"),
        align="center",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    fig.update_layout(
        title="Houses Worth Bidding On vs Not Worth It",
        xaxis_title="Actual Price (NOK)",
        yaxis_title="Predicted Price (NOK)",
        template="plotly_white",
        width=800,
        height=800,
    )

    fig.show()


if __name__ == "__main__":
    # Backend API URL
    API_URL = "http://localhost:5000"

    # Load data
    houses = load_jsonl("data/houses.jsonl")

    # Filter unsold houses
    unsold_houses = filter_unsold_houses(houses)

    for house in unsold_houses:
        print(house.get("sold"))
    if not unsold_houses:
        print("No unsold houses to analyze.")
        exit()

    # Test predictions
    y_true, y_pred, sum_savings, house_number = test_predictions(
        API_URL, unsold_houses)

    # Display average savings
    print(
        f"Sum Savings Using Model: {
            sum_savings:,.2f} NOK on {house_number} houses"
    )

    # Plot results
    if len(y_true) > 0 and len(y_pred) > 0:
        plot_worth_bidding(unsold_houses, y_true, y_pred,
                           sum_savings, house_number)
    else:
        print("No predictions could be made.")
