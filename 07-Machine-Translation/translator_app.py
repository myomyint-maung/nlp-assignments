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
@app.route('/translate', methods = ['GET','POST'])
def autocomplete():
    form = MyForm()
    name = False
    code = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        source = form.name.data 
        target = translate(source)
        form.name.data = ""
    return render_template("translate.html", form=form, name=source, code=target)

if __name__ == '__main__':
    app.run(debug=True)