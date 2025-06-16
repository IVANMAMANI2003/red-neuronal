import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path):
    df = pd.read_csv(path, sep=';')
    return df

def preprocess(df):
    # Eliminar columnas irrelevantes o redundantes
    drop_cols = ['Course', 'Nacionality', "Daytime/evening attendance\t"]
    df.drop(columns=drop_cols, inplace=True)

    # Codificar variables categóricas
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Separar features y target
    X = df.drop('Target', axis=1)
    y = df['Target']  # 0 = enrolled, 1 = dropout, 2 = graduate

    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
