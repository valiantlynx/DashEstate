import json
import random
import pickle
import base64

import pandas as pd
import numpy as np
import dash
from dash import dcc               
from dash import html as html
from dash.dependencies import Input, Output, State
import plotly.express as px




def big_front_lobe_ai_price_model(year, remodeled_year, house_color, month_number_to_marked):
    # you want a real model here that is trained and actually does something :)
    if house_color == "blue":
        return "{} NOK".format(year * 10 + remodeled_year + month_number_to_marked) 
    
    else:
        return "{} NOK".format(year * 15 + remodeled_year + month_number_to_marked) 


app = dash.Dash(__name__)
app.layout = html.Div([
    html.H3(children="KKD - Real Estate Dashboard"),
    
    html.Div([
        html.H4(children="year:"),
        dcc.Input(id='year', value='1990', type='number'),
    ]),

    html.Div([
        html.H4(children="remodeled:"),
        dcc.Input(id='remodeled', value='2015', type='number'),
    ]),

    html.Div([
        html.H4(children="color:"),
        dcc.Dropdown(['blue', 'red'], 'blue', id='color'),
    ]),

    html.Div([
        html.H4(children="Put to marked in:"),
        dcc.Dropdown(['jan', 'feb', 'march', 'april', 'november'], 'november', id='month-to-marked'),
    ]),


    html.H4(children="Predicted Price:"),
    html.Div([
        html.Pre(id="output-price")
    ])
])

@app.callback(Output('output-price', 'children'),              
              Input('year', 'value'),
              Input('remodeled', 'value'),
              Input('color', 'value'),
              Input('month-to-marked', 'value'))
def predict_price(year, remodeled, house_color, month_to_marked):    
    y = int(year)
    ry = int(remodeled)

    month_number_to_marked = 12
    if month_to_marked == "jan":
        month_number_to_marked = 1
    elif month_to_marked == "feb":
        month_number_to_marked = 2
    

    return big_front_lobe_ai_price_model(y, ry, house_color, month_number_to_marked)


if __name__ == '__main__':
    app.run_server(debug=True)






    





