# app.py
import os
import io
import json
import re
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "roberta_resume_screener_best")  # Model folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

# --- Load model/tokenizer/label encoder once on startup ---
print("Loading tokenizer and model from", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

# Load LabelEncoder safely
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# --- Job descriptions and skills mapping ---
JOB_SKILLS = {
    "Accountant": {
        "desc": "Accounting, ledger, GST/Tax filing, financial statements.",
        "skills": ["Accounting", "Tally", "GST", "Bookkeeping", "Excel", "Financial Reporting", "Taxation", "Reconciliation"]
    },
    "Advocate": {
        "desc": "Legal practice, case preparation, legal research and drafting.",
        "skills": ["Legal Research", "Drafting", "Court Representation", "Case Law", "Litigation", "Contracts"]
    },
    "Agriculture": {
        "desc": "Farming operations, crop management and agri-technology.",
        "skills": ["Crop Management", "Soil Testing", "Irrigation", "Farm Machinery", "Agronomy"]
    },
    "Apparel": {
        "desc": "Garment design, production, quality and merchandising.",
        "skills": ["Pattern Making", "Garment Construction", "Textiles", "Merchandising", "Quality Control"]
    },
    "Architecture": {
        "desc": "Design of buildings, drafting, BIM and site supervision.",
        "skills": ["AutoCAD", "Revit", "SketchUp", "BIM", "Design", "Site Supervision"]
    },
    "Arts": {
        "desc": "Creative arts, curation and studio practice.",
        "skills": ["Drawing", "Painting", "Curation", "Creative Direction", "Portfolio"]
    },
    "Automobile": {
        "desc": "Automotive engineering, servicing, OEM processes.",
        "skills": ["Auto Repair", "Vehicle Diagnostics", "CAD (CATIA/AutoCAD)", "Powertrain", "Maintenance"]
    },
    "Aviation": {
        "desc": "Aviation operations, maintenance, and ground handling.",
        "skills": ["Aviation Safety", "Aircraft Maintenance", "ATC Knowledge", "Ground Handling"]
    },
    "Banking": {
        "desc": "Bank operations, lending, compliance and risk.",
        "skills": ["KYC", "Lending", "Credit Analysis", "Compliance", "Risk Management", "MS Excel"]
    },
    "Blockchain": {
        "desc": "Smart contracts, distributed ledger and crypto.",
        "skills": ["Solidity", "Smart Contracts", "Ethereum", "Web3", "DApps", "Cryptography"]
    },
    "BPO": {
        "desc": "Customer service, process management, voice and non-voice.",
        "skills": ["Customer Service", "Call Handling", "Process Adherence", "CRM", "KPI Management"]
    },
    "Building and Construction": {
        "desc": "Construction management, civil works and site supervision.",
        "skills": ["Project Management", "Civil Engineering", "Site Supervision", "Quantity Surveying", "AutoCAD"]
    },
    "Business Analyst": {
        "desc": "Requirements gathering, stakeholder management and analytics.",
        "skills": ["Requirements Gathering", "UML", "JIRA", "Data Analysis", "SQL", "Stakeholder Management"]
    },
    "Civil Engineer": {
        "desc": "Civil design, site execution, structural principles.",
        "skills": ["Structural Design", "AutoCAD", "STAAD.Pro", "Site Management", "Reinforcement Detailing"]
    },
    "Consultant": {
        "desc": "Advisory role across business/tech domains.",
        "skills": ["Consulting", "Stakeholder Management", "Presentation", "Strategy", "Excel"]
    },
    "Data Science": {
        "desc": "Data analysis, modeling, ML and visualization.",
        "skills": ["Python", "Pandas", "scikit-learn", "TensorFlow", "PyTorch", "SQL", "Statistics", "Visualization"]
    },
    "Database": {
        "desc": "DBA and development for relational and NoSQL databases.",
        "skills": ["SQL", "MySQL", "PostgreSQL", "MongoDB", "Performance Tuning", "Backup/Restore"]
    },
    "Designing": {
        "desc": "Graphic/UI/UX design and prototyping.",
        "skills": ["Photoshop", "Illustrator", "Figma", "Sketch", "UX Research", "Prototyping"]
    },
    "DevOps": {
        "desc": "CI/CD, infra-as-code, monitoring and containerization.",
        "skills": ["Docker", "Kubernetes", "Jenkins", "Terraform", "AWS", "CI/CD", "Monitoring"]
    },
    "Digital Media": {
        "desc": "Digital marketing, social media and campaign management.",
        "skills": ["SEO", "SEM", "Google Analytics", "Social Media", "Content Creation", "Ads"]
    },
    "DotNet Developer": {
        "desc": ".NET development (C#), web services and desktop apps.",
        "skills": [".NET", "C#", "ASP.NET", "MVC", "Entity Framework", "SQL"]
    },
    "Education": {
        "desc": "Teaching, curriculum and student assessment.",
        "skills": ["Teaching", "Curriculum Development", "Lesson Planning", "Assessment", "Communication"]
    },
    "Electrical Engineering": {
        "desc": "Electrical systems, circuitry and controls.",
        "skills": ["Circuit Design", "Power Systems", "PLC", "Control Systems", "MATLAB"]
    },
    "ETL Developer": {
        "desc": "Data pipelines, transformations and warehousing.",
        "skills": ["ETL", "Informatica", "Talend", "SQL", "Data Warehousing", "Python"]
    },
    "Finance": {
        "desc": "Corporate finance, analysis, and reporting.",
        "skills": ["Financial Modelling", "Excel", "Valuation", "Budgeting", "Reporting"]
    },
    "Food and Beverages": {
        "desc": "Food production, safety, and F&B operations.",
        "skills": ["Food Safety", "Quality Control", "Operations", "HACCP"]
    },
    "Health and Fitness": {
        "desc": "Training, nutrition and wellness.",
        "skills": ["Personal Training", "Nutrition", "Wellness Coaching", "CPR"]
    },
    "Human Resources": {
        "desc": "Recruitment, payroll, employee relations.",
        "skills": ["Recruitment", "Payroll", "Labour Laws", "HRMS", "Onboarding"]
    },
    "Information Technology": {
        "desc": "IT operations, support, and services.",
        "skills": ["IT Support", "Networking", "Windows Server", "Linux", "Cloud"]
    },
    "Java Developer": {
        "desc": "Java backend/web development and frameworks.",
        "skills": ["Java", "Spring", "Hibernate", "REST", "Maven", "SQL"]
    },
    "Management": {
        "desc": "Business operations, people management, strategy.",
        "skills": ["Leadership", "Project Management", "Strategy", "Stakeholder Management"]
    },
    "Mechanical Engineer": {
        "desc": "Mechanical systems, product design and manufacturing.",
        "skills": ["CAD", "SolidWorks", "Thermodynamics", "Manufacturing", "Material Science"]
    },
    "Network Security Engineer": {
        "desc": "Network security, firewalls and monitoring.",
        "skills": ["Network Security", "Firewalls", "IDS/IPS", "VPN", "SIEM"]
    },
    "Operations Manager": {
        "desc": "Process optimization, supply chain and people management.",
        "skills": ["Process Improvement", "Supply Chain", "KPI Management", "Inventory"]
    },
    "PMO": {
        "desc": "Project governance, tracking and delivery assurance.",
        "skills": ["Project Management", "MS Project", "Scheduling", "Risk Management", "Reporting"]
    },
    "Public Relations": {
        "desc": "Media relations, communications and brand management.",
        "skills": ["PR", "Media Relations", "Content", "Press Releases", "Event Management"]
    },
    "Python Developer": {
        "desc": "Python development for web/data/automation.",
        "skills": ["Python", "Django", "Flask", "APIs", "SQL", "Unit Testing"]
    },
    "React Developer": {
        "desc": "Frontend development with React, JS and SPA architecture.",
        "skills": ["React", "JavaScript", "HTML", "CSS", "Redux", "Webpack"]
    },
    "Sales": {
        "desc": "Business development, account management and closing deals.",
        "skills": ["Sales", "Negotiation", "CRM", "Lead Generation", "Account Management"]
    },
    "SAP Developer": {
        "desc": "SAP modules, ABAP development and integration.",
        "skills": ["ABAP", "SAP FI/CO", "SAP SD", "Integration", "BAPI"]
    },
    "SQL Developer": {
        "desc": "Database development and query optimization.",
        "skills": ["SQL", "Stored Procedures", "Performance Tuning", "ETL", "Data Modeling"]
    },
    "Testing": {
        "desc": "Software QA, testing strategies and automation.",
        "skills": ["Manual Testing", "Selenium", "Test Cases", "Automation", "JIRA"]
    },
    "Web Designing": {
        "desc": "UI design, responsive layout and front-end skills.",
        "skills": ["HTML", "CSS", "JavaScript", "Responsive Design", "Figma"]
    }
}

# Build a master skill set for extraction
ALL_SKILLS = sorted({s.lower() for v in JOB_SKILLS.values() for s in v["skills"]})

# --- Helpers ---
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path_or_stream):
    text = []
    try:
        reader = PyPDF2.PdfReader(path_or_stream)
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text.append(page_text)
    except Exception as e:
        print("PDF parse error:", e)
    return "\n".join(text)

def simple_skill_extractor(text):
    text_low = text.lower()
    found = set()
    for skill in ALL_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_low):
            found.add(skill)
    return sorted(found)

def predict_label_and_confidence(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
    return pred_label, confidence, probs.tolist()

def compute_skill_overlap(resume_skills, job_skills):
    resume_set = set([s.lower() for s in resume_skills])
    job_set = set([s.lower() for s in job_skills])
    if not job_set:
        return 0.0, []
    overlap = sorted(list(resume_set & job_set))
    ratio = len(overlap) / len(job_set)
    return ratio, overlap

def combined_score(model_confidence, skill_ratio, w_model=0.6, w_skill=0.4):
    return w_model * model_confidence + w_skill * skill_ratio

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    job_list = sorted(JOB_SKILLS.keys())
    return render_template("index.html", jobs=job_list)

@app.route("/predict", methods=["POST"])
def predict():
    job = request.form.get("job")
    if not job or job not in JOB_SKILLS:
        return jsonify({"error": "Invalid job selected"}), 400

    if "resume" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(saved_path)

        # Extract resume text
        text = extract_text_from_pdf(saved_path)
        if not text:
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # Predict role
        pred_label, confidence, all_probs = predict_label_and_confidence(text)

        # Extract skills
        found_resume_skills = simple_skill_extractor(text)
        job_skills = JOB_SKILLS[job]["skills"]
        skill_ratio, overlap = compute_skill_overlap(found_resume_skills, job_skills)

        # Combined score
        score = combined_score(confidence, skill_ratio)

        # Prepare result
        result = {
            "selected_job": job,
            "job_description": JOB_SKILLS[job]["desc"],
            "required_skills": job_skills,
            "predicted_label_by_model": pred_label,
            "model_confidence": round(confidence, 4),
            "resume_skills_found": found_resume_skills,
            "skill_overlap": overlap,
            "skill_match_ratio": round(skill_ratio, 4),        # 0-1 float
            "combined_fit_score": round(score, 4),             # 0-1 float
            "all_model_probs": all_probs
        }
        return jsonify(result)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
