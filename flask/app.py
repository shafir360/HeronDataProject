from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from scripts.classifier_scripts.classifier import classifier


model_path_advanced = r'flask\models\donut_finetuned.pth'
processor_path = r'flask\models\donut_processor'
model_path_simple = r'flask\models\simpleModel'


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key
app.config['UPLOAD_FOLDER'] = 'flask/static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        
        # Handle File Upload
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Process the file as needed
            # For demonstration, we'll just return the filename
            option = request.form.get('option')
            if option == 'advanced':
                model_path = model_path_advanced
            elif option == 'simple':
                model_path = model_path_simple
            else:
                return "Incorrect Option Selected (Internal Error)"
            result = classifier(filepath,option,model_path,processor_path)
            
        else:
            flash('Invalid file type')
            return redirect(request.url)
        
        
        
        return render_template('index.html', result=result)
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
