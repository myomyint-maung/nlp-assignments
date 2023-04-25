from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from transformer import predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'

class MyForm(FlaskForm):
    source = StringField('Input python code', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/autocompleter', methods = ['GET','POST'])
def autocompleter():
    form = MyForm()
    source = False
    target = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        source = form.source.data 
        target = predict(source)
        form.source.data = ""
    return render_template('autocompleter.html', form=form, source=source, target=target)

if __name__ == '__main__':
    app.run(debug=True)