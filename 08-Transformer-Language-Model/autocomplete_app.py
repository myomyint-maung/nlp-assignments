from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from Transformer_LM import predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'

class MyForm(FlaskForm):
    name = StringField('Input code', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/autocomplete', methods = ['GET','POST'])
def autocomplete():
    form = MyForm()
    name = False
    code = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        name = form.name.data 
        code = predict(name)
        form.name.data = ""
    return render_template("autocomplete.html", form=form, name=name, code=code)

if __name__ == '__main__':
    app.run(debug=True)