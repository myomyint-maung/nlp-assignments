from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from transformers import pipeline

pipe = pipeline("text-generation", max_length=100, pad_token_id=0, eos_token_id=0, model="fairy-tale-generator-accelerate")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'

class MyForm(FlaskForm):
    prompt = StringField('Input prompt', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/generate', methods = ['GET','POST'])
def generate():
    form = MyForm()
    prompt = False
    prediction = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        prompt = form.prompt.data 
        prediction = pipe(prompt, num_return_sequences=1)[0]["generated_text"]
        form.prompt.data = ""
    return render_template('generate.html', form=form, prompt=prompt, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)