from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from translation import translate

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'

class MyForm(FlaskForm):
    name = StringField('Input Myanmar', validators=[DataRequired()])
    submit = SubmitField('Translate')

@app.route('/')
@app.route('/translator', methods = ['GET','POST'])
def translator():
    form = MyForm()
    name = False
    code = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        name = form.name.data 
        code = translate(name)
        form.name.data = ""
    return render_template("translator.html", form=form, name=name, code=code)

if __name__ == '__main__':
    app.run(debug=True)