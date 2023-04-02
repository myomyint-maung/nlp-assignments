from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from translation import translate

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'

class MyForm(FlaskForm):
    source = StringField('Input Myanmar', validators=[DataRequired()])
    submit = SubmitField('Translate')

@app.route('/')
@app.route('/translator', methods = ['GET','POST'])
def translator():
    form = MyForm()
    source = False
    target = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        source = form.source.data 
        target = translate(source)
        form.source.data = ""
    return render_template("translator.html", form=form, source=source, target=target)

if __name__ == '__main__':
    app.run(debug=True)