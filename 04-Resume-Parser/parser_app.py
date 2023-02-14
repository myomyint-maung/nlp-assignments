# Import necessary libraries
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from PyPDF2 import PdfReader 
from flask import Flask, request,render_template
from jinja2 import Template

app = Flask(__name__)

# Function to extract skills and education from a resume
def get_skills_education():

    # Load NLP model
    nlp = spacy.load('en_core_web_md')

    # Add entity ruler to NLP
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    # Load skill and education lables to the ruler
    ruler.from_disk('skills_and_education.jsonl')

    # Load a PDF file
    file = request.files("pdf_file")

    # Read the PDF file
    reader = PdfReader(file)

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

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Call the get_skills_education function and pass the pdf_file as an argument
        skills, education = get_skills_education()
        # Render the result template and pass the extracted data as arguments
        return render_template('result.html', skills=skills, education=education)
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)