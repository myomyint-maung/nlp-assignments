from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'

@app.route('/')
def index():
     return render_template('home.html')