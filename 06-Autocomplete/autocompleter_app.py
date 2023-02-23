from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from LSTM_LM import predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'

class MyForm(FlaskForm):
    prompt = StringField('Input code', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/autocomplete', methods = ['GET','POST'])
def autocomplete():
    form = MyForm()
    prompt = False
    code = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        prompt = form.prompt.data 
        code = predict(prompt)
        form.name.data = ""
    return render_template("autocomplete.html", form=form, string=string, code=code)

if __name__ == '__main__':
    app.run(debug=True)