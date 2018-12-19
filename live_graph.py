import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt;
import numpy as np
import matplotlib.animation as animation
from matplotlib import style
import time
import plotly.plotly as py
import plotly.tools as tls
import numpy as np

#style.use('ggplot')
style.use('fivethirtyeight')



fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)



def animate(i):
    graph_data = open('twitter-output.txt','r').read()
    lines = graph_data.split('\n')
    xar = []
    yar = []

    x = 0
    y = 0

    for line in lines[-300:]:
        x += 1
        if "4" in line:
            y += 1
        elif "0" in line:
            y -= 1
        xar.append(x)
        yar.append(y)

    ax1.clear()
    ax1.plot(xar, yar)
    plt.xlabel('The lastest Twitter (maximal=300)')
    plt.ylabel('Twitter Sentiment (>0:positive; <0:negative)')
    plt.title('Analyze Real-time Twitter Sentiment')

ani = animation.FuncAnimation(fig1, animate, interval=1000)
#plt.show()



fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
label = ['positive twitter', 'negative twitter', 'neutral twitter']
y_pos = np.arange(len(label))

def animate_pie(i):
    graph_data = open('twitter-output.txt','r').read()
    lines = graph_data.split('\n')
    slices = []

    n1 = 0
    n2 = 0
    n3 = 0
    for line in lines[-300:]:
        if "4" in line:
            n1 += 1
        elif "0" in line:
            n2 += 1
        elif "2" in line:
            n3 += 1
    slices.append(n1)
    slices.append(n2)
    slices.append(n3)
    #print(slices)


    ax2.clear()
    ax2.bar(y_pos, slices, align='center', alpha=0.5)
    plt.xticks(y_pos, label)
    plt.xlabel('Twitter Sentiment')
    plt.ylabel('The number of Twitter')
    plt.title('Analyze Real-time Twitter Sentiment')




slices = animation.FuncAnimation(fig2, animate_pie)



plt.show()
