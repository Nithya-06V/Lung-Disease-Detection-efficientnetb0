import json
import os
from datetime import datetime
from functools import wraps

from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS, CLASS_NAMES, CLASS_NAMES_PATH, COMPARISON_PATH, MAX_CONTENT_LENGTH, METRICS_PATH, MODEL_NAME, MODEL_PATH, REPORT_FOLDER, SECRET_KEY, UPLOAD_FOLDER
from utils.database import (
    delete_diagnosis_and_patient,
    get_all_diagnoses,
    get_all_patients,
    get_dashboard_summary,
    get_dataset_info,
    get_diagnosis_by_id,
    get_patient_by_id,
    get_patient_detail,
    get_recent_diagnoses,
    get_report_by_diagnosis_id,
    get_reports,
    get_user_by_username,
    get_disease_distribution,
    init_database,
    save_diagnosis,
    save_patient,
    save_report,
    verify_login,
)
from utils.prediction import predict_lung_disease
from utils.report_generator import generate_pdf_report

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to continue.", "danger")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model_info():
    class_names = CLASS_NAMES
    metrics = {}
    comparison = []
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            class_names = json.load(file)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH, "r", encoding="utf-8") as file:
            comparison = json.load(file).get("models", [])
    return {
        "name": MODEL_NAME,
        "technique": "Transfer Learning + Fine-Tuning",
        "framework": "TensorFlow/Keras",
        "input_size": "224 x 224",
        "classes": class_names,
        "model_file": os.path.basename(MODEL_PATH),
        "test_accuracy": metrics.get("test_accuracy"),
        "comparison": comparison,
        "ready": os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH),
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("login.html")

        try:
            authenticated = verify_login(username, password)
        except RuntimeError:
            flash("Database connection failed. Check MySQL is running and configure DB_PASSWORD before logging in.", "danger")
            return render_template("login.html")

        if authenticated:
            session["user"] = username
            flash("Login successful.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    summary = get_dashboard_summary()
    recent = get_recent_diagnoses(5)
    disease_distribution = get_disease_distribution()
    dataset = get_dataset_info()
    model_info = get_model_info()
    return render_template(
        "dashboard.html",
        summary=summary,
        recent_diagnoses=recent,
        disease_distribution=disease_distribution,
        dataset=dataset,
        model_info=model_info,
    )


@app.route("/dataset")
@login_required
def dataset():
    dataset_info = get_dataset_info()
    return render_template("dataset.html", dataset=dataset_info, class_names=dataset_info.get("classes", "").split(",") if dataset_info.get("classes") else CLASS_NAMES)


@app.route("/diagnosis", methods=["GET", "POST"])
@login_required
def diagnosis():
    if request.method == "POST":
        patient_name = request.form.get("patient_name", "").strip()
        patient_id = request.form.get("patient_id", "").strip()
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        phone = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip()
        symptoms = request.form.get("symptoms", "").strip()
        medical_history = request.form.get("medical_history", "").strip()

        if not patient_name or not patient_id or not age or not gender or not phone:
            flash("Please fill in all required patient details.", "danger")
            return render_template("diagnosis.html")

        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please select an image for diagnosis.", "danger")
            return render_template("diagnosis.html")
        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload a JPG, JPEG, or PNG image.", "danger")
            return render_template("diagnosis.html")
        if len(file.read()) > MAX_CONTENT_LENGTH:
            flash("Image file is too large. Maximum size allowed is 16MB.", "danger")
            return render_template("diagnosis.html")
        file.seek(0)

        filename = secure_filename(file.filename)
        unique_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(file_path)

        try:
            prediction = predict_lung_disease(file_path)
        except FileNotFoundError as exc:
            flash(str(exc), "danger")
            os.remove(file_path)
            return render_template("diagnosis.html")
        except Exception as exc:
            flash("Image analysis failed. Please verify the model and image format.", "danger")
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template("diagnosis.html")

        patient_data = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "age": int(age),
            "gender": gender,
            "phone": phone,
            "email": email,
            "symptoms": symptoms,
            "medical_history": medical_history,
        }

        save_patient(patient_data)
        diagnosis_id = save_diagnosis(
            patient_id,
            unique_name,
            prediction["disease"],
            float(prediction["confidence"]),
            prediction["status"],
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
        )

        session["last_result"] = {
            "diagnosis_id": diagnosis_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "age": patient_data["age"],
            "gender": gender,
            "email": email,
            "phone": phone,
            "symptoms": symptoms,
            "medical_history": medical_history,
            "image_filename": unique_name,
            "predicted_disease": prediction["disease"],
            "confidence": prediction["confidence"],
            "result_status": prediction["status"],
            "diagnosis_date": datetime.now().strftime("%Y-%m-%d"),
            "diagnosis_time": datetime.now().strftime("%H:%M:%S"),
        }

        return redirect(url_for("result", diagnosis_id=diagnosis_id))

    return render_template("diagnosis.html")


@app.route("/result/<int:diagnosis_id>")
@login_required
def result(diagnosis_id):
    diagnosis = get_diagnosis_by_id(diagnosis_id)
    if not diagnosis:
        flash("Diagnosis record not found.", "danger")
        return redirect(url_for("dashboard"))

    uploaded_image = os.path.join(app.config["UPLOAD_FOLDER"], diagnosis["image_filename"])
    diagnosis_payload = {
        "id": diagnosis_id,
        "patient_id": diagnosis["patient_id"],
        "patient_name": diagnosis["patient_name"],
        "age": diagnosis["age"],
        "gender": diagnosis["gender"],
        "email": diagnosis["email"],
        "phone": diagnosis["phone"],
        "symptoms": diagnosis["symptoms"],
        "medical_history": diagnosis["medical_history"],
        "image_filename": diagnosis["image_filename"],
        "predicted_disease": diagnosis["predicted_disease"],
        "confidence": diagnosis["confidence"],
        "result_status": diagnosis["result_status"],
        "diagnosis_date": diagnosis["diagnosis_date"],
        "diagnosis_time": diagnosis["diagnosis_time"],
    }
    return render_template("result.html", diagnosis=diagnosis_payload, image_path=uploaded_image)


@app.route("/patients")
@login_required
def patients():
    search_term = request.args.get("search", "")
    page = int(request.args.get("page", 1))
    row_data, total = get_all_patients(search_term, page, 10)
    total_pages = max(1, (total + 9) // 10)
    return render_template("patients.html", patients=row_data, total=total, page=page, total_pages=total_pages, search=search_term)


@app.route("/patient/<patient_id>")
@login_required
def patient_details(patient_id):
    patient = get_patient_detail(patient_id)
    if not patient:
        flash("Patient record not found.", "danger")
        return redirect(url_for("patients"))
    return render_template("patient_details.html", patient=patient)


@app.route("/diagnoses")
@login_required
def diagnoses():
    page = int(request.args.get("page", 1))
    rows, total = get_all_diagnoses(page, 10)
    total_pages = max(1, (total + 9) // 10)
    return render_template("diagnoses.html", diagnoses=rows, total=total, page=page, total_pages=total_pages)


@app.route("/diagnosis/<int:diagnosis_id>")
@login_required
def diagnosis_details(diagnosis_id):
    diagnosis = get_diagnosis_by_id(diagnosis_id)
    if not diagnosis:
        flash("Diagnosis not found.", "danger")
        return redirect(url_for("diagnoses"))
    return render_template("diagnoses.html", diagnosis_detail=diagnosis)


@app.route("/reports")
@login_required
def reports():
    search_term = request.args.get("search", "")
    report_rows = get_reports()
    if search_term:
        report_rows = [row for row in report_rows if search_term.lower() in str(row.get('patient_name', '')).lower() or search_term.lower() in str(row.get('patient_id', '')).lower()]
    return render_template("reports.html", reports=report_rows, search=search_term)


@app.route("/download_report/<int:diagnosis_id>")
@login_required
def download_report(diagnosis_id):
    diagnosis = get_diagnosis_by_id(diagnosis_id)
    if not diagnosis:
        flash("Invalid diagnosis ID.", "danger")
        return redirect(url_for("reports"))

    patient_data = {
        "patient_id": diagnosis["patient_id"],
        "patient_name": diagnosis["patient_name"],
        "age": diagnosis["age"],
        "gender": diagnosis["gender"],
        "phone": diagnosis["phone"],
        "email": diagnosis["email"],
        "symptoms": diagnosis["symptoms"],
        "medical_history": diagnosis["medical_history"],
    }

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], diagnosis["image_filename"])
    safe_name = f"{diagnosis['patient_id']}_Lung_Disease_Report.pdf"
    report_path = os.path.join(app.config["REPORT_FOLDER"], safe_name)

    try:
        generate_pdf_report(report_path, patient_data, diagnosis, image_path)
        save_report(diagnosis_id, safe_name)
        return send_file(report_path, as_attachment=True, download_name=safe_name)
    except Exception:
        flash("PDF generation failed. Please try again.", "danger")
        return redirect(url_for("reports"))


@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.errorhandler(404)
def page_not_found(e):
    return render_template("login.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    flash("Something went wrong on the server. Please try again later.", "danger")
    return redirect(url_for("login")), 500


if __name__ == "__main__":
    try:    
        init_database()
    except Exception:
        pass
    app.run(debug=True, host="0.0.0.0", port=5000)
