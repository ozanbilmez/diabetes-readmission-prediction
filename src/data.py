import pandas as pd
import numpy as np

def remove_outliers(df, cols):
    """
    IQR (Interquartile Range) method to remove outliers.
    """
    rows_before = len(df)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print(f"   -> Outlier Removal: Dropped {rows_before - len(df)} rows.")
    return df

def load_and_clean_diabetes_data(filepath):
    print("Loading and Advanced Processing (Diabetes Data)...")
    df = pd.read_csv(filepath, na_values='?', low_memory=False)

    # 1. Drop useless columns
    drop_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 2. Drop critical missing values
    df = df.dropna(subset=['race', 'gender', 'age', 'discharge_disposition_id'])
    
    # 3. Remove deaths (patients who died cannot be readmitted)
    df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]

    # 4. Target Engineering
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    # 5. Outlier Removal
    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                    'num_medications', 'number_diagnoses']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    if len(numeric_cols) > 0:
        df = remove_outliers(df, numeric_cols)

        # 6. Log Transform
        for col in numeric_cols:
            df[col] = np.log1p(df[col])

    # --- NEW: INTERACTION FEATURES (BOOSTING ACCURACY) ---
    # These create "severity" signals that single features miss.
    if 'num_procedures' in df.columns and 'num_medications' in df.columns:
        df['complexity_index'] = df['num_procedures'] * df['num_medications']
    
    if 'number_diagnoses' in df.columns and 'time_in_hospital' in df.columns:
        df['severity_index'] = df['number_diagnoses'] * df['time_in_hospital']

    # 7. ICD-9 Grouping
    def map_diagnosis(code):
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
        except: pass
        return 'Other'

    if 'diag_1' in df.columns:
        df['diag_1_group'] = df['diag_1'].apply(map_diagnosis)
        df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])

    # 8. Categorical Encoding
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 9. Clean up types
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        elif df[col].dtype == 'object':
            print(f"Warning: Dropping stubborn column '{col}'")
            df = df.drop(columns=[col])

    y = df['readmitted'].values
    X = df.drop(columns=['readmitted']).values.astype(float)

    print(f"Data Ready: {X.shape} samples")
    return X, y


def load_and_clean_second_dataset(filepath):
    print(f"Loading and Advanced Processing (Dataset 2: {filepath})...")
    df = pd.read_csv(filepath)

    df.columns = [c.lower() for c in df.columns]

    # 1. Target Variable
    if 'readmitted' in df.columns:
        if df['readmitted'].dtype == 'object':
            df['readmitted'] = df['readmitted'].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true', '<30'] else 0)
    else:
        print("Error: 'readmitted' column not found")
        return None, None

    # 2. Drop useless columns
    drop_cols = ['patient_id', 'encounter_id', 'unnamed: 0', 'index', 'id']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 3. Outlier Removal
    possible_numeric_cols = [
        'time_in_hospital', 'n_lab_procedures', 'n_procedures', 
        'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency',
        'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses'
    ]
    numeric_cols = [c for c in possible_numeric_cols if c in df.columns]
    
    if len(numeric_cols) > 0:
        df = remove_outliers(df, numeric_cols)

        # 4. Log Transform
        for col in numeric_cols:
            if (df[col] >= 0).all():
                df[col] = np.log1p(df[col])

    # --- NEW: INTERACTION FEATURES ---
    # Try different combinations based on available columns
    if 'n_procedures' in df.columns and 'n_medications' in df.columns:
        df['complexity_index'] = df['n_procedures'] * df['n_medications']
    elif 'num_procedures' in df.columns and 'num_medications' in df.columns:
        df['complexity_index'] = df['num_procedures'] * df['num_medications']
    
    if 'number_diagnoses' in df.columns and 'time_in_hospital' in df.columns:
        df['severity_index'] = df['number_diagnoses'] * df['time_in_hospital']

    # 5. Categorical Encoding
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 6. Final Cleanup
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        elif df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                print(f"Warning: Dropping stubborn column '{col}' in Dataset 2")
                df = df.drop(columns=[col])

    y = df['readmitted'].values
    X = df.drop(columns=['readmitted']).values.astype(float)

    print(f"Dataset 2 Ready: {X.shape} samples")
    return X, y
