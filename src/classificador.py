"""
Modulo de Classificacao do Estado do Condutor
==============================================
Implementa o modelo de Rede Neural para classificacao dos estados
do condutor conforme descrito no TCC:
- Estado de Alerta (Normal)
- Estado de Fadiga Moderada
- Estado de Sonolencia Critica

O modelo utiliza como entrada os indicadores EAR, MAR, PERCLOS e HPE,
conforme a arquitetura descrita na metodologia (Capitulo II, secao 2.4).

Arquitetura: Input(5) -> Dense(64) -> Dense(128) -> Dense(64) -> Dense(32) -> Output(3)
Ativacao: ReLU | Otimizador: Adam

Referencia: LeCun, Bengio & Hinton (2015); Goodfellow, Bengio & Courville (2016)
"""

import numpy as np
import os

import detector_fadiga.config as config

# Caminhos dos modelos treinados
MODELO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "modelo", "modelo_fadiga.pkl"
)
SCALER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "modelo", "scaler.pkl"
)


class ClassificadorFadiga:
    """
    Classificador de estados de fadiga do condutor.

    Combina um modelo de rede neural MLP (quando disponivel) com um
    classificador baseado em regras como fallback, conforme descrito no TCC.
    """

    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.usar_modelo = False

        # Tentar carregar modelo treinado
        if os.path.exists(MODELO_PATH) and os.path.exists(SCALER_PATH):
            try:
                import joblib
                self.modelo = joblib.load(MODELO_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.usar_modelo = True
                print("  [OK] Modelo de rede neural carregado com sucesso.")
            except Exception as e:
                print(f"  [AVISO] Nao foi possivel carregar o modelo: {e}")
                print("  [INFO] Utilizando classificador baseado em regras.")
        else:
            print("  [INFO] Modelo nao encontrado. Usando classificador baseado em regras.")
            print("         Execute 'python modelo/treinar_modelo.py' para treinar.")

    def preparar_features(self, metricas):
        """
        Prepara o vetor de features para entrada no modelo.

        Features: [EAR_medio, MAR, PERCLOS, pitch, roll]
        """
        features = np.array([
            metricas['ear_medio'],
            metricas['mar'],
            metricas['perclos'] / 100.0,
            abs(metricas['pitch']) / 90.0,
            abs(metricas['roll']) / 90.0
        ]).reshape(1, -1)

        return features

    def classificar_modelo(self, metricas):
        """
        Classifica o estado usando o modelo de rede neural treinado.

        Retorna: (estado, confianca)
        """
        if self.modelo is None:
            return self.classificar_regras(metricas)

        features = self.preparar_features(metricas)
        features_norm = self.scaler.transform(features)

        estado = int(self.modelo.predict(features_norm)[0])
        probabilidades = self.modelo.predict_proba(features_norm)[0]
        confianca = float(probabilidades[estado])

        return estado, confianca

    def classificar_regras(self, metricas):
        """
        Classificador baseado em regras (fallback).

        Implementa a mesma logica de decisao do modulo de decisao
        descrito na arquitetura do sistema (Figura 3.2 do TCC).

        Retorna: (estado, confianca)
        """
        ear = metricas['ear_medio']
        mar = metricas['mar']
        perclos = metricas['perclos']
        pitch = metricas['pitch']
        roll = metricas['roll']
        olhos_fechados = metricas['olhos_fechados']
        bocejando = metricas['bocejando']
        cabeca_baixa = metricas['cabeca_baixa']
        frames_fechados = metricas['frames_olhos_fechados']

        # Pontuacao de fadiga (0-100)
        score = 0.0

        # Contribuicao do EAR
        if ear < config.EAR_LIMIAR:
            score += 30
        elif ear < config.EAR_LIMIAR * 1.3:
            score += 15

        # Contribuicao do PERCLOS
        score += min(perclos * 0.8, 30)

        # Contribuicao do MAR (bocejo)
        if bocejando:
            score += 20
        elif mar > config.MAR_LIMIAR * 0.8:
            score += 10

        # Contribuicao do HPE
        if pitch > config.HPE_PITCH_LIMIAR:
            score += 15
        if abs(roll) > config.HPE_ROLL_LIMIAR:
            score += 10

        # Olhos fechados prolongados
        if frames_fechados >= config.EAR_FRAMES_CONSECUTIVOS:
            score += 25

        # Classificar com base na pontuacao
        if score >= 50:
            estado = config.ESTADO_SONOLENCIA_CRITICA
            confianca = min(score / 100, 0.99)
        elif score >= 25:
            estado = config.ESTADO_FADIGA_MODERADA
            confianca = min(score / 100, 0.85)
        else:
            estado = config.ESTADO_ALERTA
            confianca = max(1.0 - score / 100, 0.7)

        return estado, confianca

    def classificar(self, metricas):
        """
        Classifica o estado do condutor.

        Utiliza modelo de rede neural se disponivel, caso contrario usa regras.

        Retorna: (estado, confianca)
        """
        if self.usar_modelo:
            return self.classificar_modelo(metricas)
        return self.classificar_regras(metricas)
