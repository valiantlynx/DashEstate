# app.py
from flask import Flask, render_template, jsonify

app = Flask(__name__)

data = [
    {'Name': 'Property 1', 'Select': False},
    {'Name': 'Property 2', 'Select': False},
    {'Name': 'Property 3', 'Select': False},
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def get_data():
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)