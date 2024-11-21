import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = dash.Dash(__name__)
server = app.server  # the Flask instance

app.layout = html.Div([
    html.H3(children="KKD - Real Estate Dashboard", style={'textAlign': 'center', 'margin-bottom': '40px'}),

    html.Div([
        html.Label("Select Information:", style={'font-weight': 'bold', 'margin-right': '10px'}),
        dcc.Checklist(
            options=[
                {'label': 'Year', 'value': 'year'},
                {'label': 'Remodeled Year', 'value': 'remodeled_year'},
                {'label': 'Color', 'value': 'color'},
                {'label': 'Put to Market In', 'value': 'month_to_marked'}
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

@app.callback(
    Output('table-div', 'children'),
    [Input('info-checklist', 'value')]
)
def update_table(selected_checkmarks):
    logging.info(f'update_table called with: {selected_checkmarks}')

    options = ['year', 'remodeled_year', 'color', 'month_to_marked']
    labels = ['Year', 'Remodeled Year', 'Color', 'Put to Market In']

    # Determine the number of selected options
    num_selected = len(selected_checkmarks)
    num_columns = max(num_selected, 1)  # Avoid division by zero

    # Create the header row with checkboxes
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

    # Create the input row
    input_cells = []
    for option in options:
        if option in selected_checkmarks:
            if option == 'year':
                input_field = dcc.Input(id='year', value='1990', type='number', style={'width': '100%'})
            elif option == 'remodeled_year':
                input_field = dcc.Input(id='remodeled', value='2015', type='number', style={'width': '100%'})
            elif option == 'color':
                input_field = dcc.Dropdown(['blue', 'red'], 'blue', id='color', clearable=False, style={'width': '100%'})
            elif option == 'month_to_marked':
                input_field = dcc.Dropdown(
                    ['jan', 'feb', 'march', 'april', 'november'],
                    'november',
                    id='month-to-marked',
                    clearable=False,
                    style={'width': '100%'}
                )
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

    # Build the grid
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

@app.callback(
    Output('output-price', 'children'),
    Input('year', 'value'),
    Input('remodeled', 'value'),
    Input('color', 'value'),
    Input('month-to-marked', 'value'),
    Input('info-checklist', 'value')
)
def predict_price(year, remodeled, house_color, month_to_marked, selected_checkmarks):
    logging.info(
        f'predict_price called with: year={year}, remodeled={remodeled}, house_color={house_color}, month_to_marked={month_to_marked}, selected_checkmarks={selected_checkmarks}')
    y = int(year) if 'year' in selected_checkmarks and year else 0
    ry = int(remodeled) if 'remodeled_year' in selected_checkmarks and remodeled else 0
    color = house_color if 'color' in selected_checkmarks and house_color else 'blue'
    month_number_to_marked = {'jan': 1, 'feb': 2, 'march': 3, 'april': 4, 'november': 11}.get(month_to_marked, 12)

    price = big_front_lobe_ai_price_model(y, ry, color, month_number_to_marked)
    return f'Estimated Price: {price}'

def big_front_lobe_ai_price_model(year, remodeled_year, house_color, month_number_to_marked):
    logging.debug(
        f'big_front_lobe_ai_price_model called with: year={year}, remodeled_year={remodeled_year}, house_color={house_color}, month_number_to_marked={month_number_to_marked}')
    if house_color == "blue":
        price = "{} NOK".format(year * 10 + remodeled_year + month_number_to_marked)
    else:
        price = "{} NOK".format(year * 15 + remodeled_year + month_number_to_marked)
    logging.debug(f'Price calculated: {price}')
    return price

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_silence_routes_logging=True)
