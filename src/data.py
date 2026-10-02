import numpy as np
import pandas as pd

# discharge_disposition_id codes for patients who died or were discharged to hospice.
# They cannot be readmitted, so keeping them would only add easy negatives.
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

# These columns hold category codes, not quantities
ID_COLUMNS = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

DIABETES_COUNT_COLUMNS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_diagnoses', 'number_outpatient', 'number_emergency', 'number_inpatient',
]

HOSPITAL_COUNT_COLUMNS = [
    'time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications',
    'n_outpatient', 'n_inpatient', 'n_emergency',
]


def map_diagnosis(code):
    """Group an ICD-9 code into a broad disease category."""
    if str(code).startswith('250'): return 'Diabetes'
    try:
        val = float(code)
        if 390 <= val <= 459 or val == 785: return 'Circulatory'
        if 460 <= val <= 519 or val == 786: return 'Respiratory'
        if 520 <= val <= 579 or val == 787: return 'Digestive'
        if 800 <= val <= 999: return 'Injury'
        if 710 <= val <= 739: return 'Musculoskeletal'
        if 580 <= val <= 629 or val == 788: return 'Genitourinary'
        if 140 <= val <= 239: return 'Neoplasms'
    except (TypeError, ValueError):
        pass
    return 'Other'


def _one_hot(df):
    return pd.get_dummies(df, drop_first=True, dtype=float).astype(float)


def load_diabetes_130(filepath):
    """Diabetes 130-US hospitals (UCI). Returns features, target (readmitted < 30 days) and patient IDs.

    Only row-wise steps happen here. Anything that learns from the data
    (clipping bounds, scaling, feature ranking) is fitted later on the training split.
    """
    # '?' marks missing values. 'None' in the lab columns means "test not done", so pandas
    # must not turn it into NaN.
    df = pd.read_csv(filepath, na_values='?', keep_default_na=False, low_memory=False)

    df = df.drop(columns=['encounter_id', 'weight', 'payer_code', 'medical_specialty'])
    df = df.dropna(subset=['race', 'gender', 'age', 'discharge_disposition_id'])
    df = df[~df['discharge_disposition_id'].isin(EXCLUDED_DISCHARGE_IDS)].copy()

    y = (df['readmitted'] == '<30').astype(int).to_numpy()
    groups = df['patient_nbr'].to_numpy()  # used to split by patient, not a feature
    df = df.drop(columns=['readmitted', 'patient_nbr'])

    df[ID_COLUMNS] = df[ID_COLUMNS].astype(str)
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].apply(map_diagnosis)

    df[DIABETES_COUNT_COLUMNS] = np.log1p(df[DIABETES_COUNT_COLUMNS])

    # Interaction features: rough signals of how complex / severe the stay was
    df['complexity_index'] = df['num_procedures'] * df['num_medications']
    df['severity_index'] = df['number_diagnoses'] * df['time_in_hospital']

    return _one_hot(df), y, groups


def load_hospital_readmissions(filepath):
    """Hospital Readmissions (Kaggle). There are no patient IDs, so groups is None."""
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]

    y = (df['readmitted'].str.lower() == 'yes').astype(int).to_numpy()
    df = df.drop(columns=['readmitted'])

    df[HOSPITAL_COUNT_COLUMNS] = np.log1p(df[HOSPITAL_COUNT_COLUMNS])
    df['complexity_index'] = df['n_procedures'] * df['n_medications']

    return _one_hot(df), y, None
