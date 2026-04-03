import pandas as pd
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def generar_caso_de_uso_pipeline_mixto_regresion():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función pipeline_mixto_regresion(df, target_col, cat_cols, num_cols).
    """
    np.random.seed(random.randint(0, 9999))
    n = random.randint(80, 200)

    # Columnas numéricas
    num_options = [
        ('edad', lambda: np.random.randint(18, 70, n).astype(float)),
        ('antiguedad_meses', lambda: np.random.randint(1, 72, n).astype(float)),
        ('num_dispositivos', lambda: np.random.randint(1, 6, n).astype(float)),
        ('ingresos_mensual', lambda: np.random.uniform(500, 5000, n).round(2)),
    ]
    n_num = random.randint(2, 4)
    selected_num = random.sample(num_options, n_num)
    num_cols = [x[0] for x in selected_num]

    # Columnas categóricas
    cat_options = [
        ('plan', ['basico', 'medio', 'premium']),
        ('ciudad', ['Medellin', 'Bogota', 'Cali', 'Barranquilla']),
        ('operador', ['Claro', 'Movistar', 'Tigo']),
    ]
    n_cat = random.randint(1, 2)
    selected_cat = random.sample(cat_options, n_cat)
    cat_cols = [x[0] for x in selected_cat]

    target_col = 'consumo_gb'

    # Construir DataFrame
    data = {}
    for name, gen in selected_num:
        vals = gen()
        # Introducir ~5% NaN
        mask = np.random.choice([True, False], size=n, p=[0.05, 0.95])
        vals = vals.astype(float)
        vals[mask] = np.nan
        data[name] = vals

    for name, cats in selected_cat:
        vals = np.random.choice(cats, size=n).astype(object)
        mask = np.random.choice([True, False], size=n, p=[0.03, 0.97])
        vals[mask] = np.nan
        data[name] = vals

    # Target: combinación lineal de features numéricas + ruido
    target = np.zeros(n)
    for i, (name, _) in enumerate(selected_num):
        clean = np.where(np.isnan(data[name]), np.nanmedian(data[name]), data[name])
        target += (i + 1) * 0.5 * clean
    target += np.random.normal(0, 5, n)
    target = np.abs(target).round(2)
    data[target_col] = target

    df = pd.DataFrame(data)

    # INPUT
    input_data = {
        'df': df.copy(),
        'target_col': target_col,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
    }

    # OUTPUT esperado
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
        ],
        remainder='drop'
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)
    r2_test = float(pipeline.score(X_test, y_test))
    n_features_out = int(preprocessor.fit_transform(X_train).shape[1])

    output_data = {
        'pipeline': pipeline,
        'r2_test': r2_test,
        'n_features_out': n_features_out,
    }

    return input_data, output_data


if __name__ == '__main__':
    entrada, salida_esperada = generar_caso_de_uso_pipeline_mixto_regresion()
    print("=== INPUT ===")
    print(f"target_col: {entrada['target_col']}")
    print(f"cat_cols: {entrada['cat_cols']}")
    print(f"num_cols: {entrada['num_cols']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    print("\n=== OUTPUT ESPERADO ===")
    print(f"r2_test: {salida_esperada['r2_test']:.4f}")
    print(f"n_features_out: {salida_esperada['n_features_out']}")
