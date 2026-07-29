from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/load')
def load():
    return render_template('load.html')

@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)