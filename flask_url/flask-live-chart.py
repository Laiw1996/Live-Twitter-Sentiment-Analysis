import json
from time import time
import random
from flask import Markup
from flask import Flask, render_template, make_response

app = Flask(__name__)


colors = [
    "#F7464A", "#46BFBD", "#FDB45C"]


@app.route('/')
def hello_world():
    return render_template('index.html', data='test')

@app.route('/bar-chart')
def bar():
    graph_data = open('twitter-output.txt','r').read()
    lines = graph_data.split('\n')
    values = []
    labels = ['positive twitter', 'negative twitter', 'neutral twitter']
    n1 = 0
    n2 = 0
    n3 = 0
    for line in lines[-300:]:
        if "2" in line:
            n1 += 1
        elif "0" in line:
            n2 += 1
        elif "1" in line:
            n3 += 1
    values.append(n1)
    values.append(n2)
    values.append(n3)

    bar_labels=labels
    bar_values=values
    return render_template('bar-chart.html', title='Bitcoin Monthly Price in USD', max=500, labels=bar_labels, values=bar_values)



@app.route('/line-chart')
def line():
    graph_data = open('twitter-output.txt','r').read()
    lines = graph_data.split('\n')
    values = []
    labels = []

    x = 0
    y = 0

    for line in lines[-300:]:
        x += 1
        if "2" in line:
            y += 1
        elif "0" in line:
            y -= 1
        labels.append(x)
        values.append(y)
    
    line_labels=labels
    line_values=values
    return render_template('line-chart.html', title='Bitcoin Monthly Price in USD', max=100, labels=line_labels, values=line_values)


@app.route('/pie-chart')
def pie():
    graph_data = open('twitter-output.txt','r').read()
    lines = graph_data.split('\n')
    labels=['POSITVE','NEGATIVE','NEUTURAL']
    values=[]
    posNum=0
    negNum=0
    neuNum=0
    for line in lines[-700:]:
        if "2" in line:
           posNum+=1
        elif "0" in line:
            negNum+=1
        elif "1" in line:
            neuNum+=1
    values.append(posNum)
    values.append(negNum)
    values.append(neuNum)

    pie_labels = labels
    pie_values = values
    return render_template('pie-chart.html', title='Bitcoin Monthly Price in USD', max=500, set=zip(values, labels, colors))



@app.route('/live-data')
def live_data():
    # Create a PHP array and echo it as JSON
    graph_data = open('twitter-output.txt','r').read()
    lines = graph_data.split('\n')
    xar = []
    x = 0
    for line in lines[-300:]:
        if "2" in line:
            x=2
        elif "0" in line:
            x=0
        elif "1" in line:
            x=1
        xar.append(x)
       
    data = [time()*1000, random.choice(xar)]
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
