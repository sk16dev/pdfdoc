from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import PyPDF2
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'document' not in request.files:
        return redirect(request.url)
    file = request.files['document']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('document_page', filename=filename))

@app.route('/document/<filename>')
def document_page(filename):
    return render_template('document.html', filename=filename)

@app.route('/get_answer', methods=['POST'])
def get_answer():
    filename = request.form['filename']
    question = request.form['question']

    
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

    
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_text = "question: {} context: {}".format(question, text)
    input_ids = tokenizer.encode(input_text, truncation=True, max_length=512, return_tensors='pt')
    output_ids = model.generate(input_ids, max_new_tokens=50)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return render_template('answer.html', question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
