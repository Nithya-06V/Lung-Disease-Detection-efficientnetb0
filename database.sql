CREATE DATABASE IF NOT EXISTS lung_disease_db;
USE lung_disease_db;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL UNIQUE,
    patient_name VARCHAR(150) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(20) NOT NULL,
    phone VARCHAR(30) NOT NULL,
    email VARCHAR(150),
    symptoms TEXT,
    medical_history TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS diagnoses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    image_filename VARCHAR(255) NOT NULL,
    predicted_disease VARCHAR(150) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    result_status VARCHAR(50) NOT NULL,
    diagnosis_date DATE NOT NULL,
    diagnosis_time TIME NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dataset_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    description TEXT,
    image_count INT,
    number_of_classes INT,
    image_format VARCHAR(50),
    image_size VARCHAR(50),
    training_images INT,
    validation_images INT,
    testing_images INT,
    classes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    diagnosis_id INT NOT NULL,
    report_filename VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id) ON DELETE CASCADE
);

INSERT INTO users (username, password)
VALUES ('admin', 'admin123')
ON DUPLICATE KEY UPDATE username = 'admin';

INSERT INTO dataset_info (
    dataset_name,
    description,
    image_count,
    number_of_classes,
    image_format,
    image_size,
    training_images,
    validation_images,
    testing_images,
    classes
)
SELECT
    'Lung Disease Image Dataset',
    'Chest X-ray image collection used for lung disease classification including Normal, Pneumonia, and Fibrosis examples. Update this table when the dataset changes.',
    0,
    3,
    'PNG',
    '224x224',
    0,
    0,
    0,
    'Normal,Pneumonia,Fibrosis'
WHERE NOT EXISTS (
    SELECT 1
    FROM dataset_info
    WHERE dataset_name = 'Lung Disease Image Dataset'
);
