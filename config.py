"""
Configuracoes do Sistema de Deteccao de Fadiga e Sonolencia
============================================================
Parametros visuais e comportamentais conforme descrito no TCC:
- EAR (Eye Aspect Ratio)
- MAR (Mouth Aspect Ratio)
- PERCLOS (Percentage of Eye Closure)
- HPE (Head Pose Estimation)
"""

# ============================================================
# LIMIARES DE DETECCAO
# ============================================================

# EAR - Eye Aspect Ratio
EAR_LIMIAR = 0.16              # Limiar para considerar olho fechado
EAR_FRAMES_CONSECUTIVOS = 15   # Frames consecutivos (legado)
EAR_TEMPO_FECHADO = 2.0        # Segundos com olhos fechados para alerta

# MAR - Mouth Aspect Ratio
MAR_LIMIAR = 0.55              # Limiar para considerar boca aberta (bocejo)
MAR_TEMPO_BOCEJO = 0.7         # Segundos que a boca deve permanecer aberta
MAR_BOCEJOS_ALERTA = 3         # Numero de bocejos para disparar alerta
MAR_RESET_TEMPO = 10           # Segundos para resetar contador de bocejos
MAR_DEBOUNCE = 2.0             # Segundos entre contagem de bocejos

# PERCLOS - Percentage of Eye Closure
PERCLOS_JANELA = 30            # Janela temporal em segundos
PERCLOS_LIMIAR_MODERADO = 15   # % para fadiga moderada
PERCLOS_LIMIAR_CRITICO = 30    # % para sonolencia critica

# HPE - Head Pose Estimation
HPE_PITCH_LIMIAR = 20          # Graus de inclinacao para baixo (pitch)
HPE_YAW_LIMIAR = 30            # Graus de rotacao lateral (yaw)
HPE_ROLL_LIMIAR = 20           # Graus de inclinacao lateral (roll)
HPE_TEMPO_ALERTA = 2.0         # Segundos de inclinacao para alerta

# ============================================================
# ESTADOS DO CONDUTOR
# ============================================================
ESTADO_ALERTA = 0              # Condutor alerta
ESTADO_FADIGA_MODERADA = 1     # Fadiga moderada
ESTADO_SONOLENCIA_CRITICA = 2  # Sonolencia critica

NOMES_ESTADO = {
    ESTADO_ALERTA: "NORMAL",
    ESTADO_FADIGA_MODERADA: "FADIGA MODERADA",
    ESTADO_SONOLENCIA_CRITICA: "SONOLENCIA CRITICA"
}

CORES_ESTADO = {
    ESTADO_ALERTA: (0, 200, 0),            # Verde
    ESTADO_FADIGA_MODERADA: (0, 200, 255),  # Amarelo/Laranja
    ESTADO_SONOLENCIA_CRITICA: (0, 0, 255)  # Vermelho
}

# ============================================================
# SISTEMA DE ALERTA
# ============================================================
ALERTA_BEEP_FREQ = 1500        # Frequencia do beep em Hz
ALERTA_BEEP_DURACAO = 800      # Duracao do beep em ms
ALERTA_INTERVALO = 1.0         # Intervalo entre alertas em segundos

# ============================================================
# CAMERA E PROCESSAMENTO
# ============================================================
CAMERA_ID = 0                  # ID da camera (0 = webcam padrao)
CAMERA_LARGURA = 1280          # Largura desejada do frame
CAMERA_ALTURA = 720            # Altura desejada do frame
FPS_ALVO = 30                  # FPS alvo do sistema

# ============================================================
# LANDMARKS DO MEDIAPIPE FACE MESH
# ============================================================

# Olho Direito (6 pontos para calculo do EAR)
OLHO_DIREITO = {
    'p1': 33,   # Canto externo
    'p2': 160,  # Palpebra superior (externo)
    'p3': 158,  # Palpebra superior (interno)
    'p4': 133,  # Canto interno
    'p5': 153,  # Palpebra inferior (interno)
    'p6': 144   # Palpebra inferior (externo)
}

# Olho Esquerdo (6 pontos para calculo do EAR)
OLHO_ESQUERDO = {
    'p1': 362,  # Canto externo
    'p2': 385,  # Palpebra superior (externo)
    'p3': 387,  # Palpebra superior (interno)
    'p4': 263,  # Canto interno
    'p5': 373,  # Palpebra inferior (interno)
    'p6': 380   # Palpebra inferior (externo)
}

# Boca (pontos para calculo do MAR)
BOCA = {
    'esquerda': 61,
    'direita': 291,
    'superior_1': 13,
    'superior_2': 82,
    'superior_3': 312,
    'inferior_1': 14,
    'inferior_2': 87,
    'inferior_3': 317
}

# Pontos para HPE (Head Pose Estimation)
HPE_PONTOS = {
    'nariz': 1,
    'queixo': 152,
    'olho_esquerdo': 33,
    'olho_direito': 263,
    'boca_esquerda': 61,
    'boca_direita': 291
}

# ============================================================
# MODELO CNN
# ============================================================
CNN_INPUT_FEATURES = 5         # EAR_medio, MAR, PERCLOS, pitch, roll
CNN_NUM_CLASSES = 3            # Alerta, Fadiga Moderada, Sonolencia Critica
CNN_MODELO_PATH = "modelo/modelo_fadiga.pkl"
CNN_SCALER_PATH = "modelo/scaler.pkl"
