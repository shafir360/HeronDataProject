# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from scripts.classifier_scripts.classifier import classifier

# Define paths to your models and processor
model_path_advanced = r'src\models\donut_finetuned.pth'
processor_path = r'src\models\donut_processor'
model_path_simple = r'src\models\simpleModel'

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure key in production

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'flask/static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handle the main page rendering and form submission.
    """
    result = None
    if request.method == 'POST':
        option = request.form.get('option')
        
        if option == 'advanced':
            # Handle File Upload for Advanced Classifier
            if 'file' not in request.files:
                flash('No file part in the form.')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No file selected for uploading.')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                # Call the classifier with the file path
                result = classifier(filepath, option, model_path_advanced, processor_path)
            else:
                flash('Invalid file type. Allowed types are txt, pdf, png, jpg, jpeg.')
                return redirect(request.url)
        
        elif option == 'simple':
            # Handle Text Input for Simple Classifier
            text_input = request.form.get('text_input')
            if not text_input or text_input.strip() == '':
                flash('No text input provided.')
                return redirect(request.url)
            # Call the classifier with the text input
            result = classifier(text_input, option, model_path_simple)
        
        else:
            flash('Incorrect option selected. Please try again.')
            return redirect(request.url)
        
        return render_template('index.html', result=result)
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Run the Flask app in debug mode (disable in production)
    app.run(debug=True)
