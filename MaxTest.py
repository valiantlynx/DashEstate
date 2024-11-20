from flask import Flask, render_template, jsonify, request
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import json
import logging

app = Flask(__name__)

# Load data from a JSON Lines file
data = []
with open('data/houses.jsonl', 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Extract unique values from data
unique_values = {
    "colors": list(set(item.get("color", "unknown") for item in data)),
    "fireplace": list(set(item.get("fireplace", "unknown") for item in data)),
    "parking": list(set(item.get("parking", "unknown") for item in data)),
    "sold_in_month": list(set(item.get("sold_in_month", "unknown") for item in data))
}

# Calculate maximum values for numeric inputs
maximums = {
    "year": max(item.get("year", 0) for item in data),
    "remodeled": max(item.get("remodeled", 0) for item in data),
    "bathrooms": max(item.get("bathrooms", 0) for item in data),
    "days_on_marked": max(item.get("days_on_marked", 0) for item in data),
    "external_storage_m2": max(item.get("external_storage_m2", 0) for item in data),
    "kitchens": max(item.get("kitchens", 0) for item in data),
    "lot_w": max(item.get("lot_w", 0) for item in data),
    "price": max(item.get("price", 0) for item in data),
    "rooms": max(item.get("rooms", 0) for item in data),
    "storage_rating": max(item.get("storage_rating", 0) for item in data),
    "sun_factor": max(item.get("sun_factor", 0) for item in data)
}


@app.route('/')
def index():
    return render_template('index.html', unique_values=unique_values, maximums=maximums)


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json

    # Handling possible None or empty string values
    year = int(input_data.get('year', 0) or 0)
    remodeled = int(input_data.get('remodeled', 0) or 0)
    color = input_data.get('color', 'blue')
    bathrooms = int(input_data.get('bathrooms', 0) or 0)
    days_on_marked = float(input_data.get('days_on_marked', 0) or 0)
    external_storage_m2 = float(input_data.get('external_storage_m2', 0) or 0)
    kitchens = int(input_data.get('kitchens', 0) or 0)
    lot_w = float(input_data.get('lot_w', 0) or 0)
    price = float(input_data.get('price', 0) or 0)
    rooms = int(input_data.get('rooms', 0) or 0)
    storage_rating = int(input_data.get('storage_rating', 0) or 0)
    sun_factor = float(input_data.get('sun_factor', 0) or 0)

    # Predict price using a mock function. Adjust the function to include other variables as needed.
    predicted_price = big_front_lobe_ai_price_model(year, remodeled, color, bathrooms, days_on_marked,
                                                    external_storage_m2,
                                                    kitchens, lot_w, price, rooms, storage_rating, sun_factor)
    return jsonify({'price': predicted_price})


def big_front_lobe_ai_price_model(year, remodeled_year, house_color, bathrooms, days_on_marked, external_storage_m2,
                                  kitchens, lot_w, price, rooms, storage_rating, sun_factor):
    computed_price = year * 10 + remodeled_year + bathrooms + days_on_marked + external_storage_m2 + kitchens + lot_w \
                     + price + rooms + storage_rating + sun_factor
    price_str = "{} NOK".format(computed_price)
    logging.debug(f'Price calculated: {price_str}')
    return price_str


# Dash app integration
logging.basicConfig(level=logging.DEBUG)
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/')

dash_app.layout = html.Div([
    html.H3(children="KKD - Real Estate Dashboard", style={'textAlign': 'center', 'margin-bottom': '40px'}),

    html.Div([
        html.Label("Select Information:", style={'font-weight': 'bold', 'margin-right': '10px'}),
        dcc.Checklist(
            options=[
                {'label': 'Year', 'value': 'year'},
                {'label': 'Remodeled Year', 'value': 'remodeled'},
                {'label': 'Color', 'value': 'color'},
                {'label': 'Bathrooms', 'value': 'bathrooms'},
                {'label': 'Days on Market', 'value': 'days_on_marked'},
                {'label': 'External Storage (m2)', 'value': 'external_storage_m2'},
                {'label': 'Kitchens', 'value': 'kitchens'},
                {'label': 'Lot Width', 'value': 'lot_w'},
                {'label': 'Price', 'value': 'price'},
                {'label': 'Rooms', 'value': 'rooms'},
                {'label': 'Storage Rating', 'value': 'storage_rating'},
                {'label': 'Sun Factor', 'value': 'sun_factor'}
            ],
            id='info-checklist',
            inline=True,
            labelStyle={'margin-right': '15px'}
        ),
    ], style={'textAlign': 'center', 'margin-bottom': '20px'}),

    html.Div(id='table-div', style={'width': '80%', 'margin': '0 auto'}),

    html.H4(children="Predicted Price:", style={'textAlign': 'center', 'margin-top': '40px'}),
    html.Div([
        html.Pre(id="output-price", style={'textAlign': 'center', 'font-size': '24px', 'font-weight': 'bold'})
    ])
])


@dash_app.callback(
    Output('table-div', 'children'),
    [Input('info-checklist', 'value')]
)
def update_table(selected_checkmarks):
    options = ['year', 'remodeled', 'color', 'bathrooms', 'days_on_marked', 'external_storage_m2',
               'kitchens', 'lot_w', 'price', 'rooms', 'storage_rating', 'sun_factor']
    labels = ['Year', 'Remodeled Year', 'Color', 'Bathrooms', 'Days on Market', 'External Storage (m2)',
              'Kitchens', 'Lot Width', 'Price', 'Rooms', 'Storage Rating', 'Sun Factor']
    num_selected = len(selected_checkmarks)
    num_columns = max(num_selected, 1)

    header_cells = []
    for option, label in zip(options, labels):
        is_checked = option in selected_checkmarks
        header_cells.append(html.Div([
            dcc.Checklist(
                options=[{'label': label, 'value': option}],
                value=[option] if is_checked else [],
                id=f'{option}-checkbox',
                inline=True,
                labelStyle={'display': 'block', 'textAlign': 'center'}
            )
        ], style={
            'border': '1px solid grey',
            'padding': '10px',
            'textAlign': 'center',
            'width': 'auto',
            'flex': '1' if is_checked else '0.5',
            'display': 'block'
        }))

    input_cells = []
    for option in options:
        if option in selected_checkmarks:
            if option == 'year':
                input_field = dcc.Input(id='year', value='1990', type='number', min=0,
                                        style={'width': '100%'})
            elif option == 'remodeled':
                input_field = dcc.Input(id='remodeled', value='2015', type='number', min=-1,
                                        style={'width': '100%'})
            elif option == 'color':
                input_field = dcc.Dropdown(unique_values['colors'], unique_values['colors'][0], id='color',
                                           clearable=False, style={'width': '100%'})
            elif option == 'bathrooms':
                input_field = dcc.Input(id='bathrooms', value=1, type='number', min=0,
                                        style={'width': '100%'})
            elif option == 'days_on_marked':
                input_field = dcc.Input(id='days_on_marked', value=0, type='number', min=0,
                                        style={'width': '100%'})
            elif option == 'external_storage_m2':
                input_field = dcc.Input(id='external_storage_m2', value=0, type='number', min=0,
                                        style={'width': '100%'})
            elif option == 'kitchens':
                input_field = dcc.Input(id='kitchens', value=1, type='number', min=0,
                                        style={'width': '100%'})
            elif option == 'lot_w':
                input_field = dcc.Input(id='lot_w', value=0, type='number', min=0,
                                        style={'width': '100%'})
            elif option == 'price':
                input_field = dcc.Input(id='price', value=0, type='number', min=0,
                                        style={'width': '100%'})
            elif option == 'rooms':
                input_field = dcc.Input(id='rooms', value=1, type='number', min=0,
                                        style={'width': '100%'})
            elif option == 'storage_rating':
                input_field = dcc.Input(id='storage_rating', value=1, type='number', min=0,
                                        style={'width': '100%'})
            elif option == 'sun_factor':
                input_field = dcc.Input(id='sun_factor', value=0, type='number', min=0,
                                        style={'width': '100%'})

            input_cells.append(html.Div(input_field, style={
                'border': '1px solid grey',
                'padding': '10px',
                'width': 'auto',
                'flex': '1',
                'display': 'block'
            }))
        else:
            input_cells.append(html.Div(style={
                'border': '1px solid grey',
                'padding': '10px',
                'width': 'auto',
                'flex': '0.5',
                'display': 'block'
            }))

    table = html.Div([
        html.Div(header_cells, style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'margin-bottom': '5px'
        }),
        html.Div(input_cells, style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        })
    ])

    return table


@dash_app.callback(
    Output('output-price', 'children'),
    Input('year', 'value'),
    Input('remodeled', 'value'),
    Input('color', 'value'),
    Input('bathrooms', 'value'),
    Input('days_on_marked', 'value'),
    Input('external_storage_m2', 'value'),
    Input('kitchens', 'value'),
    Input('lot_w', 'value'),
    Input('price', 'value'),
    Input('rooms', 'value'),
    Input('storage_rating', 'value'),
    Input('sun_factor', 'value'),
    Input('info-checklist', 'value')
)
def predict_price(year, remodeled, house_color, bathrooms, days_on_marked, external_storage_m2,
                  kitchens, lot_w, price, rooms, storage_rating, sun_factor, selected_checkmarks):
    y = int(year) if 'year' in selected_checkmarks and year else 0
    ry = int(remodeled) if 'remodeled' in selected_checkmarks and remodeled else 0
    color = house_color if 'color' in selected_checkmarks and house_color else 'blue'
    b = int(bathrooms) if 'bathrooms' in selected_checkmarks and bathrooms else 0
    dom = float(days_on_marked) if 'days_on_marked' in selected_checkmarks and days_on_marked else 0
    es = float(external_storage_m2) if 'external_storage_m2' in selected_checkmarks and external_storage_m2 else 0
    k = int(kitchens) if 'kitchens' in selected_checkmarks and kitchens else 0
    lw = float(lot_w) if 'lot_w' in selected_checkmarks and lot_w else 0
    p = float(price) if 'price' in selected_checkmarks and price else 0
    r = int(rooms) if 'rooms' in selected_checkmarks and rooms else 0
    sr = int(storage_rating) if 'storage_rating' in selected_checkmarks and storage_rating else 0
    sf = float(sun_factor) if 'sun_factor' in selected_checkmarks and sun_factor else 0

    predicted_price = big_front_lobe_ai_price_model(y, ry, color, b, dom, es, k, lw, p, r, sr, sf)
    return f'Estimated Price: {predicted_price}'


if __name__ == '__main__':
    app.run(debug=True)
