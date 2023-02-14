from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from PyPDF2 import PdfReader

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_uploaded_resume():
     file_list = os.listdir('static/files')
     for filename in file_list:
          os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def get_skills_education(file_path):

    # Load NLP model
    nlp = spacy.load('en_core_web_md')

    # Add entity ruler to NLP
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    # Load skill and education lables to the ruler
    ruler.from_disk('static/data/skills_and_education.jsonl')

    # Load the PDF file
    reader = PdfReader(file_path)

    # Extract text from the PDF
    text = str()

    for i in range(len(reader.pages)):
        page = reader.pages[i]
    
        if i == 0:
            text += page.extract_text()
        else:
            text += ' ' + page.extract_text()
    
    # Preprocess the text    
    stopwords = list(STOP_WORDS)
    doc = nlp(text)
    cleaned_tokens = []
    
    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and \
            token.pos_ != 'SPACE' and token.pos_ != 'SYM':
                cleaned_tokens.append(token.text.lower().strip())
    
    text = " ".join(cleaned_tokens)

    # Extract sklls and education
    doc = nlp(text)
    skills = []
    education = []

    for i in range(len(doc.ents)):
        
        if doc.ents[i].label_ == "SKILL":
            skills.append(doc.ents[i].text)
        
        if doc.ents[i].label_ == "EDUCATION_PRO":
            education.append(doc.ents[i].text)
        
        if doc.ents[i].label_ == "EDUCATION_OF" and doc.ents[i+1].label_ == "SKILL" and doc.ents[i+2].label_ == "SKILL":
            education.append(doc.ents[i].text + ' of ' + doc.ents[i+1].text + ' in ' + doc.ents[i+2].text)
        elif doc.ents[i].label_ == "EDUCATION_OF" and doc.ents[i+1].label_ == "SKILL" and doc.ents[i+2].label_ != "SKILL":
            education.append(doc.ents[i].text + ' of ' + doc.ents[i+1].text)
        
        if doc.ents[i].label_ == "EDUCATION_IN" and doc.ents[i+1].label_ == "SKILL":
            education.append(doc.ents[i].text + ' in ' + doc.ents[i+1].text)

    skills = list(set(skills))
    education = list(set(education))

    skills.sort()
    education.sort()
    education.reverse()

    return skills, education

@app.route('/')
@app.route('/home')
def home():
    remove_uploaded_resume()
    return render_template('home.html')

@app.route('/upload_file', methods=["POST"])
def upload_file():
     remove_uploaded_resume()
     if 'File' not in request.files:
          flash('No file uploaded')
          return redirect('/')

     file_upload = request.files['File']
     file_upload.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file_upload.filename)))
     return redirect('/parsed_file')

@app.route('/parsed_file', methods=["GET"])
def parsed_file():
     filename = os.listdir('static/files')

     if len(filename) == 0:
          return redirect('/')

     skills, education = get_skills_education('./static/files/' + filename[0])
     
     return render_template('result.html', skills=skills, education=education)

if __name__ == '__main__':
    app.run(debug=True)