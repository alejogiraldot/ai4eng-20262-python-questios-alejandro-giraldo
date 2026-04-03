import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def generar_caso_de_uso_evaluar_clasificador_multiclase():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función evaluar_clasificador_multiclase(X, y_raw, test_size, random_state).
    """
    seed = random.randint(0, 9999)
    np.random.seed(seed)

    n_samples = random.randint(150, 400)
    n_features = random.randint(5, 15)
    n_informative = random.randint(3, min(n_features, 8))
    test_size = round(random.choice([0.2, 0.25, 0.3]), 2)
    random_state = random.randint(0, 100)

    # Clases multiclase para productos industriales
    all_class_sets = [
        ['sin_defecto', 'defecto_leve', 'defecto_grave'],
        ['aprobado', 'revision', 'rechazado'],
        ['optimo', 'aceptable', 'critico'],
    ]
    clases = random.choice(all_class_sets)
    n_classes = len(clases)

    # Generar datos
    X, y_int = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(1, n_features - n_informative - 2),
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed
    )
    y_raw = [clases[i] for i in y_int]

    # INPUT
    input_data = {
        'X': X.copy(),
        'y_raw': y_raw.copy(),
        'test_size': test_size,
        'random_state': random_state,
    }

    # OUTPUT esperado
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    acc = float(accuracy_score(y_test, y_pred))
    precisions = precision_score(y_test, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_test, y_pred, average=None, zero_division=0)

    por_clase = {}
    for idx, class_name in enumerate(le.classes_):
        por_clase[class_name] = {
            'precision': float(precisions[idx]),
            'recall': float(recalls[idx]),
        }

    output_data = {
        'accuracy': acc,
        'por_clase': por_clase,
        'label_encoder': le,
    }

    return input_data, output_data


if __name__ == '__main__':
    entrada, salida_esperada = generar_caso_de_uso_evaluar_clasificador_multiclase()
    print("=== INPUT ===")
    print(f"X shape: {entrada['X'].shape}")
    print(f"Clases únicas en y_raw: {set(entrada['y_raw'])}")
    print(f"test_size: {entrada['test_size']}")
    print(f"random_state: {entrada['random_state']}")
    print("\n=== OUTPUT ESPERADO ===")
    print(f"accuracy: {salida_esperada['accuracy']:.4f}")
    print("por_clase:")
    for clase, metricas in salida_esperada['por_clase'].items():
        print(f"  {clase}: {metricas}")
