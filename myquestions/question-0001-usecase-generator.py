import pandas as pd
import numpy as np
import random


def generar_caso_de_uso_transformar_a_largo():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función transformar_a_largo(df, id_col, columnas_semana).
    """
    # 1. Configuración aleatoria
    n_pacientes = random.randint(4, 12)
    semanas_posibles = [0, 2, 4, 6, 8, 12, 16, 24]
    n_semanas = random.randint(2, min(5, len(semanas_posibles)))
    semanas_elegidas = sorted(random.sample(semanas_posibles, n_semanas))

    # 2. Construir el DataFrame ancho
    id_col = 'paciente_id'
    columnas_semana = [f'semana_{s}' for s in semanas_elegidas]

    data = {id_col: list(range(1, n_pacientes + 1))}
    for col in columnas_semana:
        # Valores biomarcadores: floats con algunos NaN ocasionales (~5%)
        vals = np.random.uniform(1.0, 10.0, size=n_pacientes)
        mask = np.random.choice([True, False], size=n_pacientes, p=[0.05, 0.95])
        vals = vals.astype(float)
        vals[mask] = np.nan
        data[col] = vals

    df = pd.DataFrame(data)

    # 3. Construir el INPUT
    input_data = {
        'df': df.copy(),
        'id_col': id_col,
        'columnas_semana': columnas_semana,
    }

    # 4. Calcular el OUTPUT esperado
    df_largo = pd.melt(
        df,
        id_vars=[id_col],
        value_vars=columnas_semana,
        var_name='semana',
        value_name='valor'
    )
    df_largo['semana'] = df_largo['semana'].str.extract(r'(\d+)').astype(int)

    resumen = (
        df_largo.groupby('semana')['valor']
        .agg(promedio='mean', desviacion_std='std')
        .reset_index()
        .sort_values('semana')
        .reset_index(drop=True)
    )

    output_data = resumen

    return input_data, output_data


if __name__ == '__main__':
    entrada, salida_esperada = generar_caso_de_uso_transformar_a_largo()
    print("=== INPUT ===")
    print(f"id_col: {entrada['id_col']}")
    print(f"columnas_semana: {entrada['columnas_semana']}")
    print("DataFrame:")
    print(entrada['df'].head())
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
