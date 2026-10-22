from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHARTS_DIR = PROJECT_ROOT / "app" / "static" / "charts"

RAW_DATA = DATA_DIR / "placement_data.csv"
PRE_TRAIN = DATA_DIR / "preprocessed_train.csv"
PRE_TEST = DATA_DIR / "preprocessed_test.csv"
PIPELINE = MODELS_DIR / "pipeline.pkl"
DEFAULT_MODEL = MODELS_DIR / "default_model.pkl"
METRICS = MODELS_DIR / "metrics.json"
SUMMARY = MODELS_DIR / "summary.json"

TARGET = "PlacementStatus"
DROPS = ["StudentID", "IsAnomaly", "Salary Package"]

NUMERICS = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4", "SGPA_Sem5",
    "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8", "CGPA", "AttendancePercent",
    "Internships", "Projects", "Workshops", "Certifications",
    "Publications", "AptitudeTestScore", "SoftSkillsRating",
    "CodingTestScore", "MockInterviewScore", "ExtraCurricular",
    "CollegeTier", "CGPA_Tier",
]

CATEGORICALS = [
    "Gender", "City", "Stream", "Specialisation", "Hostel",
    "HistoryOfBacklogs",
]

ORDERED_AS_NUMERIC = ["CollegeTier", "CGPA_Tier"]

PALETTE = ["#2563eb", "#0ea5e9", "#f59e0b", "#10b981", "#8b5cf6", "#64748b"]