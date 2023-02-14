from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return "<b><i>Resume Parser<i></b> by Myo Myint Maung"

if __name__ == '__main__':
    app.run(debug==True)