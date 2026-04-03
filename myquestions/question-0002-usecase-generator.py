import pandas as pd
import numpy as np
import random


def generar_caso_de_uso_construir_features_vibracion():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función construir_features_vibracion(df, motor_col, fecha_col, vibracion_col).
    """
    # 1. Configuración aleatoria
    n_motores = random.randint(1, 3)
    dias_por_motor = random.randint(10, 20)

    motor_col = 'motor'
    fecha_col = 'fecha'
    vibracion_col = 'vibracion'

    # 2. Construir DataFrame
    filas = []
    inicio = pd.Timestamp('2024-01-01') + pd.Timedelta(days=random.randint(0, 30))
    for m in range(1, n_motores + 1):
        fechas = pd.date_range(inicio, periods=dias_por_motor)
        vibraciones = np.random.uniform(0.5, 8.0, size=dias_por_motor).round(3)
        for f, v in zip(fechas, vibraciones):
            filas.append({motor_col: f'M{m}', fecha_col: f, vibracion_col: v})

    df = pd.DataFrame(filas)

    # 3. Construir INPUT
    input_data = {
        'df': df.copy(),
        'motor_col': motor_col,
        'fecha_col': fecha_col,
        'vibracion_col': vibracion_col,
    }

    # 4. Calcular OUTPUT esperado
    df_work = df.copy()
    df_work[fecha_col] = pd.to_datetime(df_work[fecha_col])
    df_work = df_work.sort_values([motor_col, fecha_col]).reset_index(drop=True)

    df_work['lag_1'] = df_work.groupby(motor_col)[vibracion_col].shift(1)
    df_work['lag_7'] = df_work.groupby(motor_col)[vibracion_col].shift(7)
    df_work['tendencia_3d'] = (
        df_work[vibracion_col] - df_work.groupby(motor_col)[vibracion_col].shift(3)
    )

    df_work = df_work.dropna(subset=['lag_1', 'lag_7', 'tendencia_3d'])
    df_work = df_work.reset_index(drop=True)

    output_data = df_work

    return input_data, output_data


if __name__ == '__main__':
    entrada, salida_esperada = generar_caso_de_uso_construir_features_vibracion()
    print("=== INPUT ===")
    print(f"motor_col: {entrada['motor_col']}")
    print(f"fecha_col: {entrada['fecha_col']}")
    print(f"vibracion_col: {entrada['vibracion_col']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    print("\n=== OUTPUT ESPERADO (primeras 5 filas) ===")
    print(salida_esperada.head())
    print(f"Shape: {salida_esperada.shape}")
