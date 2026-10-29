# Lung Disease Image Detection and Diagnosis System

## Project Overview
This project is a professional Flask-based web application designed for AI-assisted lung disease image screening. It allows an administrator to log in, upload chest X-ray or lung images, run a trained Convolutional Neural Network (CNN), view prediction results, store records in MySQL, and download patient diagnosis reports.

## Features
- Secure admin login using Flask sessions
- Professional Bootstrap 5 dashboard and responsive UI
- Image upload with validation and preview
- CNN prediction using TensorFlow/Keras
- MySQL storage for patients, diagnoses, dataset metadata, and reports
- PDF report generation for each diagnosis
- Disease distribution dashboard using Chart.js
- Dataset information page with dynamic metadata
- Patient and diagnosis history pages
- Downloadable reports and structured admin workflow

## Technologies Used
- Python
- Flask
- TensorFlow/Keras
- MySQL Workbench / MySQL
- Bootstrap 5
- Chart.js
- ReportLab
- Pillow
- Werkzeug
- Jinja2

## Requirements
- Python 3.10+
- MySQL Server and MySQL Workbench
- VS Code
- Internet access for package installation

## Installation
1. Open VS Code.
2. Open the project folder.
3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the environment on Windows:

```bash
venv\Scripts\activate
```

If PowerShell blocks activation:

```bash
venv\Scripts\activate.bat
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

## MySQL Workbench Setup
1. Open MySQL Workbench.
2. Create a local connection.
3. Create a database named "lung_disease_db".
4. Import the SQL file: `database.sql`.
5. Ensure the database user has privileges to create tables and insert data.

## Database Configuration
Update the values in `config.py`:

```python
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "your_mysql_password"
DB_NAME = "lung_disease_db"
```

Use environment variables if needed:

```bash
set DB_HOST=localhost
set DB_USER=root
set DB_PASSWORD=your_mysql_password
set DB_NAME=lung_disease_db
```

## Dataset Structure
Your folders should look like this:

```text
dataset/
    train/
        Normal/
        Pneumonia/
        Fibrosis/
    validation/
        Normal/
        Pneumonia/
        Fibrosis/
    test/
        Normal/
        Pneumonia/
        Fibrosis/
```

The current workspace includes folders such as `Normal`, `Pneumonia`, and `Fibrosis` under the dataset root. Update the classes in `config.py` and the dataset metadata in `database.sql` to match the actual model.

## CNN Model Training
Run:

```bash
python train_model.py
```

This script builds a simple CNN using ImageDataGenerator and saves the trained model as:

```text
model/lung_disease_model.keras
```

It also prints:
- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Confusion matrix
- Classification report

## Model Placement
Place the trained model here:

```text
lung_disease_flask/model/lung_disease_model.keras
```

If using a custom filename, update `MODEL_PATH` in `config.py`.

## Running the Flask Application
From the project folder:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Login Credentials
- Username: `admin`
- Password: `admin123`

## Project Screenshots Placeholder
Add screenshots here after running the app:
- Login page
- Dashboard
- Diagnosis page
- Result page
- Patient records
- Reports page

## Troubleshooting
### MySQL connection error
- Confirm MySQL is running.
- Check username/password settings in `config.py`.
- Ensure `database.sql` has been imported.

### Model not found
- Ensure the trained `.keras` file exists in `model/`.
- Check `MODEL_PATH` in `config.py`.

### Invalid image upload
- Use JPG, JPEG, or PNG files only.
- Ensure file size does not exceed 16MB.

### Import/packaging issues
- Reinstall dependencies:

```bash
pip install -r requirements.txt
```

## Medical Disclaimer
This system is an AI-based screening and educational tool. It is not a substitute for professional medical diagnosis, consultation, or treatment decisions.

---

## Important Notes
- The demo uses a simple beginner-friendly CNN approach.
- The project is designed to work with a real trained model, not random predictions.
- Update the dataset metadata and class names when using a different dataset or model architecture.




<img width="1917" height="906" alt="image" src="https://github.com/user-attachments/assets/84cc33c6-a415-4238-ade2-a4488b9cc8f6" />
