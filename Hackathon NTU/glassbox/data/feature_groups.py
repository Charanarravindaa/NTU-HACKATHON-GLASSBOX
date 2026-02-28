# Chunk assignment definitions for UCI Heart Disease (Cleveland) dataset
# 13 features total, grouped by clinical semantics

FEATURE_NAMES = [
    'age', 'sex', 'cp',          # Chunk A: Demographics
    'trestbps', 'chol', 'thalach', # Chunk B: Cardiovascular Vitals
    'fbs', 'restecg', 'oldpeak', 'slope',  # Chunk C: Lab & Diagnostic
    'ca', 'thal', 'exang',        # Chunk D: Structural Findings
]

CHUNK_GROUPS = {
    'Demographics': {
        'features': ['age', 'sex', 'cp'],
        'indices': [0, 1, 2],
        'color': '#4CC9F0',
        'description': 'Patient baseline risk factors',
    },
    'Vitals': {
        'features': ['trestbps', 'chol', 'thalach'],
        'indices': [3, 4, 5],
        'color': '#F72585',
        'description': 'Cardiovascular physiological state',
    },
    'LabDiagnostic': {
        'features': ['fbs', 'restecg', 'oldpeak', 'slope'],
        'indices': [6, 7, 8, 9],
        'color': '#7209B7',
        'description': 'Clinical lab and diagnostic measurements',
    },
    'Structural': {
        'features': ['ca', 'thal', 'exang'],
        'indices': [10, 11, 12],
        'color': '#3A0CA3',
        'description': 'Anatomical and structural findings',
    },
}

GHOST_GATES = [
    ('Demographics', 'Vitals', 'Demographics→Vitals'),
    ('Vitals', 'LabDiagnostic', 'Vitals→LabDiagnostic'),
    ('LabDiagnostic', 'Structural', 'LabDiag→Structural'),
    ('Demographics', 'LabDiagnostic', 'Demographics→LabDiag'),
]

FEATURE_RANGES = {
    'age':      (29, 77),
    'sex':      (0, 1),
    'cp':       (0, 3),
    'trestbps': (94, 200),
    'chol':     (126, 564),
    'thalach':  (71, 202),
    'fbs':      (0, 1),
    'restecg':  (0, 2),
    'oldpeak':  (0.0, 6.2),
    'slope':    (0, 2),
    'ca':       (0, 3),
    'thal':     (0, 3),
    'exang':    (0, 1),
}

FEATURE_DESCRIPTIONS = {
    'age':      'Age (years)',
    'sex':      'Sex (0=F, 1=M)',
    'cp':       'Chest Pain Type (0-3)',
    'trestbps': 'Resting Blood Pressure (mmHg)',
    'chol':     'Serum Cholesterol (mg/dl)',
    'thalach':  'Max Heart Rate Achieved',
    'fbs':      'Fasting Blood Sugar >120 (0/1)',
    'restecg':  'Resting ECG Result (0-2)',
    'oldpeak':  'ST Depression Induced by Exercise',
    'slope':    'Slope of Peak Exercise ST (0-2)',
    'ca':       'Number of Major Vessels (0-3)',
    'thal':     'Thalassemia (0-3)',
    'exang':    'Exercise Induced Angina (0/1)',
}
