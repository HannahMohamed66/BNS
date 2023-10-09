from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the data
    df = pd.read_csv('./BNS.csv', encoding='latin-1')

    # Training the model
    X = df[['PSR name', 'Mp (M)®']]
    y = df['Target']
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # Extract input from the form
    psr_name = request.form['psr_name']
    mp = float(request.form['mp'])

    # Make a prediction
    prediction = clf.predict([[psr_name, mp]])

    return render_template('result.html', prediction=prediction)

@app.route('/plot')
def plot():
    # Load the data
    df = pd.DataFrame({
        'Mp (M)&#174;': [2.2, 1.97, 1.62, 1.616, 1.3, 1.4, 1.908, 2.01, 2.072, 1.3381,
                    1.56, 1.3452, 1.4, 1.56, 1.4, 1.365, 1.4398, 1.358, 1.358, 1.76],
        'Target': ['YES', 'YES', 'YES', 'YES', 'YES', 'NO', 'NO', 'YES', 'YES', 'YES',
                   'NO', 'YES', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'YES']
    })

    df_first_20 = df.head(20)
    x = df_first_20['Mp (M)&#174;']
    y = df_first_20['Target']

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel('Mp (M)&#174;')
    ax.set_ylabel('Target')
    ax.set_title('Scatter Plot')

    # Save the plot to a file
    plot_filename = 'plot.png'
    plt.savefig(plot_filename)

    # Create a second plot
    fig2, ax2 = plt.subplots()
    ax2.scatter(x, y)
    ax2.set_xlabel('Mp (M)&#174;')
    ax2.set_ylabel('Target')

    # Return the rendered template with the plot filename
    return render_template('plot.html', plot_filename=plot_filename)

if __name__ == '__main__':
    app.run(debug=True)