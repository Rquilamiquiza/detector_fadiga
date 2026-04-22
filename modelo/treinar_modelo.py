"""
Script de Treinamento do Modelo de Rede Neural
================================================
Treina um modelo de Rede Neural Multicamada (MLP) para classificacao
dos estados de fadiga do condutor.

Classificacao em 3 estados:
- 0: Alerta (Normal)
- 1: Fadiga Moderada (Bocejo/Cansaco)
- 2: Sonolencia Critica (Olhos Fechados)

Conforme descrito no Capitulo II (secao 2.4) e Capitulo III do TCC.

O modelo MLP (Multi-Layer Perceptron) utiliza a mesma arquitetura
de camadas densas com ativacao ReLU descrita para a CNN, adequada
para classificacao baseada em features extraidas (EAR, MAR, PERCLOS, HPE).

Metricas de avaliacao:
- Acuracia
- Precisao
- Sensibilidade (Recall)
- F-Score
- Matriz de Confusao

Uso:
    python modelo/treinar_modelo.py
"""

import numpy as np
import os
import sys
import time

# Adicionar diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detector_fadiga.config as config


def gerar_dados_sinteticos(n_amostras=1500):
    """
    Gera dados sinteticos para treinamento e validacao do modelo.

    Em um cenario real, estes dados seriam obtidos a partir da coleta
    de imagens descrita na secao 2.2 do TCC, com rotulagem baseada
    na Karolinska Sleepiness Scale.

    Features: [EAR_medio, MAR, PERCLOS (normalizado), pitch (norm), roll (norm)]
    Labels: 0 (Alerta), 1 (Fadiga Moderada), 2 (Sonolencia Critica)
    """
    np.random.seed(42)
    dados = []
    labels = []

    amostras_por_classe = n_amostras // 3

    # --- ESTADO ALERTA ---
    # Valores calibrados com MediaPipe (EAR real ~0.18-0.28 com olhos abertos)
    for _ in range(amostras_por_classe):
        ear = np.random.normal(0.23, 0.04)
        mar = np.random.normal(0.12, 0.06)
        perclos = np.random.normal(0.03, 0.02)
        pitch = np.random.normal(0.03, 0.04)
        roll = np.random.normal(0.02, 0.03)

        dados.append([
            np.clip(ear, 0.10, 0.45),
            np.clip(mar, 0.0, 0.4),
            np.clip(perclos, 0.0, 0.10),
            np.clip(abs(pitch), 0.0, 0.15),
            np.clip(abs(roll), 0.0, 0.12)
        ])
        labels.append(0)

    # --- FADIGA MODERADA ---
    # Bocejos frequentes, PERCLOS elevado, olhos semi-cerrados
    for _ in range(amostras_por_classe):
        ear = np.random.normal(0.18, 0.04)
        mar = np.random.normal(0.55, 0.15)
        perclos = np.random.normal(0.22, 0.06)
        pitch = np.random.normal(0.18, 0.08)
        roll = np.random.normal(0.10, 0.06)

        dados.append([
            np.clip(ear, 0.08, 0.30),
            np.clip(mar, 0.2, 1.0),
            np.clip(perclos, 0.12, 0.40),
            np.clip(abs(pitch), 0.0, 0.5),
            np.clip(abs(roll), 0.0, 0.4)
        ])
        labels.append(1)

    # --- SONOLENCIA CRITICA ---
    # Olhos fechados, PERCLOS muito alto, cabeca caida
    for _ in range(amostras_por_classe):
        ear = np.random.normal(0.10, 0.04)
        mar = np.random.normal(0.18, 0.10)
        perclos = np.random.normal(0.50, 0.12)
        pitch = np.random.normal(0.35, 0.10)
        roll = np.random.normal(0.22, 0.10)

        dados.append([
            np.clip(ear, 0.0, 0.18),
            np.clip(mar, 0.0, 1.0),
            np.clip(perclos, 0.25, 1.0),
            np.clip(abs(pitch), 0.15, 1.0),
            np.clip(abs(roll), 0.0, 1.0)
        ])
        labels.append(2)

    return np.array(dados, dtype=np.float32), np.array(labels, dtype=np.int32)


def treinar():
    """
    Treina o modelo de rede neural e salva os pesos.

    Utiliza MLPClassifier do scikit-learn com arquitetura equivalente
    a CNN descrita no TCC:
    - Camadas ocultas: 64 -> 128 -> 64 -> 32 neuronios
    - Ativacao: ReLU
    - Otimizador: Adam
    - 50 iteracoes (epocas)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_score, recall_score, f1_score
    )
    import joblib

    print("=" * 60)
    print("  TREINAMENTO DO MODELO - DETECCAO DE FADIGA")
    print("  Rede Neural Multicamada (MLP)")
    print("=" * 60)

    # ============================================================
    # 1. GERAR CONJUNTO DE DADOS
    # ============================================================
    print("\n[1/6] Gerando conjunto de dados...")
    X, y = gerar_dados_sinteticos(n_amostras=1500)
    print(f"   Total de amostras: {len(X)}")
    print(f"   Features por amostra: {X.shape[1]}")
    print(f"   Features: [EAR, MAR, PERCLOS, Pitch, Roll]")
    print(f"   Classes: 0=Alerta, 1=Fadiga Moderada, 2=Sonolencia Critica")
    print(f"   Distribuicao: {np.bincount(y)}")

    # ============================================================
    # 2. DIVIDIR DADOS (70% treino, 15% validacao, 15% teste)
    # ============================================================
    print("\n[2/6] Dividindo conjunto de dados...")
    X_treino, X_temp, y_treino, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_teste, y_val, y_teste = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"   Treino:    {len(X_treino)} amostras ({len(X_treino)/len(X)*100:.0f}%)")
    print(f"   Validacao: {len(X_val)} amostras ({len(X_val)/len(X)*100:.0f}%)")
    print(f"   Teste:     {len(X_teste)} amostras ({len(X_teste)/len(X)*100:.0f}%)")

    # ============================================================
    # 3. NORMALIZAR DADOS
    # ============================================================
    print("\n[3/6] Normalizando dados (StandardScaler)...")
    scaler = StandardScaler()
    X_treino_norm = scaler.fit_transform(X_treino)
    X_val_norm = scaler.transform(X_val)
    X_teste_norm = scaler.transform(X_teste)
    print("   Dados normalizados com media 0 e desvio padrao 1.")

    # ============================================================
    # 4. CRIAR E TREINAR MODELO
    # ============================================================
    print("\n[4/6] Criando arquitetura da Rede Neural...")
    print("   Arquitetura: Input(5) -> Dense(64) -> Dense(128) -> Dense(64) -> Dense(32) -> Output(3)")
    print("   Ativacao: ReLU")
    print("   Otimizador: Adam")
    print("   Regularizacao: alpha=0.001")

    modelo = MLPClassifier(
        hidden_layer_sizes=(64, 128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        verbose=True
    )

    print("\n   Treinando modelo...\n")
    inicio = time.time()
    modelo.fit(X_treino_norm, y_treino)
    tempo_treino = time.time() - inicio

    print(f"\n   Treinamento concluido em {tempo_treino:.1f} segundos")
    print(f"   Epocas executadas: {modelo.n_iter_}")
    print(f"   Loss final: {modelo.loss_:.4f}")

    # ============================================================
    # 5. AVALIAR NO CONJUNTO DE VALIDACAO
    # ============================================================
    print("\n[5/6] Avaliando no conjunto de validacao...")
    y_val_pred = modelo.predict(X_val_norm)
    acc_val = accuracy_score(y_val, y_val_pred)
    print(f"   Acuracia validacao: {acc_val * 100:.1f}%")

    # ============================================================
    # 6. AVALIAR NO CONJUNTO DE TESTE
    # ============================================================
    print("\n[6/6] Avaliando desempenho no conjunto de teste...")
    print("=" * 60)

    y_pred = modelo.predict(X_teste_norm)

    nomes_classes = ["Alerta", "Fadiga Moderada", "Sonolencia Critica"]

    # Relatorio completo
    print("\n  RELATORIO DE CLASSIFICACAO:")
    print("-" * 55)
    print(classification_report(y_teste, y_pred, target_names=nomes_classes,
                                 digits=3))

    # Matriz de confusao
    print("  MATRIZ DE CONFUSAO:")
    print("-" * 55)
    cm = confusion_matrix(y_teste, y_pred)
    print(f"                      Predito")
    print(f"                  Alerta  Fadiga  Sonolencia")
    for i, nome in enumerate(["Alerta    ", "Fadiga    ", "Sonolencia"]):
        print(f"  Real {nome}  {cm[i][0]:5d}   {cm[i][1]:5d}   {cm[i][2]:5d}")

    # Metricas globais (conforme Tabela 3 do TCC)
    acuracia = accuracy_score(y_teste, y_pred)
    precisao = precision_score(y_teste, y_pred, average='weighted')
    sensibilidade = recall_score(y_teste, y_pred, average='weighted')
    f_score = f1_score(y_teste, y_pred, average='weighted')

    print("\n" + "=" * 60)
    print("  METRICAS DE DESEMPENHO DO SISTEMA")
    print("  (Conforme Tabela 3 do TCC)")
    print("=" * 60)
    print(f"  Acuracia:              {acuracia * 100:.1f}%")
    print(f"  Precisao:              {precisao * 100:.1f}%")
    print(f"  Sensibilidade (Recall): {sensibilidade * 100:.1f}%")
    print(f"  F-Score:               {f_score * 100:.1f}%")
    print("=" * 60)

    # Resultados por classe (conforme Tabela 2 do TCC)
    print("\n  RESULTADOS POR CLASSE:")
    print("-" * 55)
    for i, nome in enumerate(nomes_classes):
        total = np.sum(y_teste == i)
        corretos = cm[i][i]
        erros = total - corretos
        taxa = corretos / total * 100 if total > 0 else 0
        print(f"  {nome:22s} | Amostras: {total:3d} | "
              f"Corretas: {corretos:3d} | Erros: {erros:2d} | "
              f"Taxa: {taxa:.1f}%")

    # ============================================================
    # SALVAR MODELO E SCALER
    # ============================================================
    caminho_modelo = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "modelo_fadiga.pkl"
    )
    caminho_scaler = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scaler.pkl"
    )

    joblib.dump(modelo, caminho_modelo)
    joblib.dump(scaler, caminho_scaler)

    print(f"\n  Modelo salvo em: {caminho_modelo}")
    print(f"  Scaler salvo em: {caminho_scaler}")

    print("\n" + "=" * 60)
    print("  TREINAMENTO CONCLUIDO COM SUCESSO!")
    print("=" * 60)

    return modelo, scaler


if __name__ == "__main__":
    treinar()
