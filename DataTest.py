from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import json

# Load the houses JSONL file
houses_file_path = './data/houses.jsonl'
houses = []

# Read the JSONL file
with open(houses_file_path, 'r') as file:
    for i, line in enumerate(file):
        house_data = json.loads(line)
        house_data['id'] = i + 1  # Add house ID based on the line number
        houses.append(house_data)

# Sorting houses by ID for consistency
houses_sorted = sorted(houses, key=lambda x: x.get('id', 0))

# Available logical parameters for plotting
logical_params = ["advertisement", "agent_id", "color", "district_id", "fireplace", "lot_w", "parking",
                  "bathrooms", "condition_rating", "remodeled", "size", "days_on_marked", "external_storage_m2",
                  "storage_rating", "sun_factor", "year", "sold", "sold_in_month", "kitchens"]

# Define categorical parameters that should be visualized using bar or pie charts
categorical_params = ["advertisement", "color", "fireplace", "remodeled", "parking", "sold", "sold_in_month", "district_id", "agent_id"]

app = Dash(__name__)

# Custom CSS for grid layout (3 columns of graphs per row)
app.layout = html.Div([
    html.H4('Price vs Logical Parameters'),
    dcc.Dropdown(
        id='param-selector',
        options=[{'label': param.replace('_', ' ').title(), 'value': param} for param in logical_params],
        value=logical_params + categorical_params,  # Default to show all parameters
        clearable=True,
        multi=True,  # Enable multi-select
        placeholder="Select parameters to display"
    ),

    # Price slider, only shown for pie charts
    html.Div(id='price-slider-container', children=[
        html.P("Select Price Range:"),
        dcc.RangeSlider(
            id='price-slider',
            min=min(house['price'] for house in houses_sorted),
            max=max(house['price'] for house in houses_sorted),
            value=[min(house['price'] for house in houses_sorted), max(house['price'] for house in houses_sorted)],
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'display': 'none'}),  # Initially hidden

    # Container for dynamically generated graphs in a grid layout
    html.Div(id="graph-container", style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around'}),

])

@app.callback(
    [Output("graph-container", "children"), Output("price-slider-container", "style")],
    [Input("param-selector", "value"), Input("price-slider", "value")]
)
def update_dashboard(params, price_range):
    if not params:
        return [], {'display': 'none'}

    # Remove duplicates from the selected parameters
    unique_params = list(dict.fromkeys(params))

    graphs = []
    price_slider_style = {'display': 'none'}

    for param in unique_params:
        if param == "remodeled":
            # Handle 'remodeled' field where -1 means 'Not Remodeled'
            filtered_houses = [house for house in houses_sorted if house.get("remodeled") is not None]
            for house in filtered_houses:
                house["remodeled_status"] = "Remodeled" if house["remodeled"] != -1 else "Not Remodeled"

            fig = px.pie(
                filtered_houses, names="remodeled_status", title="Remodeled vs Not Remodeled"
            )
        elif param in ["district_id", "agent_id", "lot_w"]:
            # Handle categorical parameters like district_id and agent_id
            valid_houses = [house for house in houses_sorted if house.get(param)]
            if len(valid_houses) > 0:
                fig = px.bar(
                    valid_houses, x=param, y="price",
                    labels={param: param.replace('_', ' ').title(), "price": "Price"},
                    title=f"Price vs {param.replace('_', ' ').title()}"
                )
            else:
                fig = px.bar(
                    houses_sorted, x=param, y="price",
                    labels={param: param.replace('_', ' ').title(), "price": "Price"},
                    title=f"Price vs {param.replace('_', ' ').title()} (No Data)"
                )
        elif param not in categorical_params:
            # Generate scatter plot for continuous parameters
            fig = px.scatter(
                houses_sorted, x=param, y="price",
                labels={param: param.replace('_', ' ').title(), "price": "Price"},
                title=f"Price vs {param.replace('_', ' ').title()}"
            )
        else:
            # Show the price slider only if any categorical parameter is selected
            price_slider_style = {'display': 'block'}

            # Filter data by price range for categorical parameters
            filtered_houses = [house for house in houses_sorted if price_range[0] <= house['price'] <= price_range[1]]

            if param in ["sold", "fireplace", "parking"]:  # For binary yes/no-like fields
                fig = px.pie(
                    filtered_houses, names=param, title=f"Distribution of {param.replace('_', ' ').title()} (Filtered by Price)"
                )
            else:
                fig = px.bar(
                    filtered_houses, x=param, y="price",
                    labels={param: param.replace('_', ' ').title(), "price": "Price"},
                    title=f"Price vs {param.replace('_', ' ').title()} (Filtered by Price)"
                )

        # Add the generated figure into a new Graph component
        graphs.append(html.Div(dcc.Graph(figure=fig, id=f"graph-{param}"), style={'width': '32%', 'padding': '10px'}))

    return graphs, price_slider_style


if __name__ == '__main__':
    app.run_server(debug=True)
