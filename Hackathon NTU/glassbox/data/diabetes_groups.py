"""
Feature group definitions for Pima Indians Diabetes dataset.

8 features split into 3 semantically meaningful chunks:
  Chunk A — Metabolic (Glucose, Insulin, BMI): core diabetes biomarkers
  Chunk B — Physiological (BloodPressure, SkinThickness, DiabetesPedigree): body measurements
  Chunk C — Demographic (Pregnancies, Age): patient background

Ghost gates model cross-chunk interactions, e.g. how age/pregnancy history
modulates the interpretation of metabolic markers.
"""

DIABETES_FEATURE_NAMES = [
    'Pregnancies',       # 0
    'Glucose',           # 1
    'BloodPressure',     # 2
    'SkinThickness',     # 3
    'Insulin',           # 4
    'BMI',               # 5
    'DiabetesPedigreeFunction',  # 6
    'Age',               # 7
]

DIABETES_CHUNK_GROUPS = {
    'Metabolic':      {'indices': [1, 4, 5], 'color': '#06b6d4'},   # Glucose, Insulin, BMI
    'Physiological':  {'indices': [2, 3, 6], 'color': '#8b5cf6'},   # BP, SkinThick, Pedigree
    'Demographic':    {'indices': [0, 7],    'color': '#f59e0b'},    # Pregnancies, Age
}
