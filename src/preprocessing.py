import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()

    df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    return df

