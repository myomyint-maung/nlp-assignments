from flask import Flask, render_template, redirect
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class UploadFile(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/')
@app.route('/home')
def home():
    form = UploadFile()
    return render_template('home.html', form=form)

@app.route('/uploadfile', methods=['POST'])
def uploadfile():
    form = UploadFile()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
    return redirect('/parsefile')

@app.route('/parsefile', methods=['GET'])
def display():
    filename = os.listdir('static/files')
    if len(filename) == 0:
        return redirect('/home')
    
    skills, education = get_skills_education('./static/files/' + filename[0])
    
    return render_template('result.html', skills=skills, education=education)

if __name__ == '__main__':
    app.run(debug=True)