"""
Modulo de Deteccao de Fadiga e Sonolencia
==========================================
Implementa os indicadores visuais e comportamentais descritos no TCC:
- EAR (Eye Aspect Ratio) - Soukupova & Cech, 2016
- MAR (Mouth Aspect Ratio) - Abbas et al., 2022
- PERCLOS (Percentage of Eye Closure) - Wierwille et al., 1994
- HPE (Head Pose Estimation) - Murphy-Chutorian & Trivedi, 2009

Utiliza MediaPipe Face Mesh para extracao dos 468 landmarks faciais.
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time
from collections import deque

import detector_fadiga.config as config


class DetectorFadiga:
    """Classe principal para deteccao de fadiga e sonolencia do condutor."""

    def __init__(self):
        # Inicializa MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # --- Variaveis EAR ---
        self.ear_esquerdo = 0.0
        self.ear_direito = 0.0
        self.ear_medio = 0.0
        self.frames_olhos_fechados = 0
        self.olhos_fechados = False
        self.olhos_fechados_inicio = 0

        # --- Variaveis MAR ---
        self.mar = 0.0
        self.boca_aberta = False
        self.boca_aberta_inicio = 0
        self.bocejo_contador = 0
        self.ultimo_bocejo = time.time()
        self.bocejando = False

        # --- Variaveis PERCLOS ---
        self.perclos = 0.0
        self.historico_ear = deque(maxlen=config.PERCLOS_JANELA * config.FPS_ALVO)
        self.fps_real = config.FPS_ALVO
        self.ultimo_frame_time = time.time()

        # --- Variaveis HPE ---
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.cabeca_baixa = False
        self.cabeca_baixa_inicio = 0

        # --- Warm-up ---
        self.frames_processados = 0
        self.warmup_completo = False
        self.WARMUP_FRAMES = config.FPS_ALVO * 2  # 2 segundos

        # --- Estado geral ---
        self.estado_atual = config.ESTADO_ALERTA
        self.face_detectada = False
        self.ultimo_face_landmarks = None

        # Pontos 3D do modelo facial para HPE (modelo generico)
        self.pontos_3d_modelo = np.array([
            (0.0, 0.0, 0.0),          # Nariz
            (0.0, -330.0, -65.0),      # Queixo
            (-225.0, 170.0, -135.0),   # Olho esquerdo
            (225.0, 170.0, -135.0),    # Olho direito
            (-150.0, -150.0, -125.0),  # Boca esquerda
            (150.0, -150.0, -125.0)    # Boca direita
        ], dtype=np.float64)

    def _calcular_distancia(self, p1, p2):
        """Calcula a distancia euclidiana entre dois pontos 2D."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _obter_ponto(self, landmarks, idx, w, h):
        """Converte landmark normalizado para coordenadas de pixel."""
        lm = landmarks.landmark[idx]
        return (int(lm.x * w), int(lm.y * h))

    def _calcular_ear(self, landmarks, pontos_olho, w, h):
        """
        Calcula o Eye Aspect Ratio (EAR) conforme Soukupova & Cech (2016).

        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

        Onde p1-p6 sao os 6 landmarks ao redor do olho.
        """
        p1 = self._obter_ponto(landmarks, pontos_olho['p1'], w, h)
        p2 = self._obter_ponto(landmarks, pontos_olho['p2'], w, h)
        p3 = self._obter_ponto(landmarks, pontos_olho['p3'], w, h)
        p4 = self._obter_ponto(landmarks, pontos_olho['p4'], w, h)
        p5 = self._obter_ponto(landmarks, pontos_olho['p5'], w, h)
        p6 = self._obter_ponto(landmarks, pontos_olho['p6'], w, h)

        # Distancias verticais
        dist_vertical_1 = self._calcular_distancia(p2, p6)
        dist_vertical_2 = self._calcular_distancia(p3, p5)

        # Distancia horizontal
        dist_horizontal = self._calcular_distancia(p1, p4)

        if dist_horizontal == 0:
            return 0.0

        ear = (dist_vertical_1 + dist_vertical_2) / (2.0 * dist_horizontal)
        return ear

    def _calcular_mar(self, landmarks, w, h):
        """
        Calcula o Mouth Aspect Ratio (MAR) conforme Abbas et al. (2022).

        MAR = (||p14 - p18|| + ||p15 - p17||) / (2 * ||p13 - p19||)

        Utiliza pontos de referencia dos labios superior, inferior e laterais.
        """
        boca = config.BOCA

        esq = self._obter_ponto(landmarks, boca['esquerda'], w, h)
        dir_ = self._obter_ponto(landmarks, boca['direita'], w, h)
        sup1 = self._obter_ponto(landmarks, boca['superior_1'], w, h)
        sup2 = self._obter_ponto(landmarks, boca['superior_2'], w, h)
        sup3 = self._obter_ponto(landmarks, boca['superior_3'], w, h)
        inf1 = self._obter_ponto(landmarks, boca['inferior_1'], w, h)
        inf2 = self._obter_ponto(landmarks, boca['inferior_2'], w, h)
        inf3 = self._obter_ponto(landmarks, boca['inferior_3'], w, h)

        # Distancias verticais da boca
        dist_v1 = self._calcular_distancia(sup1, inf1)
        dist_v2 = self._calcular_distancia(sup2, inf2)
        dist_v3 = self._calcular_distancia(sup3, inf3)

        # Distancia horizontal
        dist_h = self._calcular_distancia(esq, dir_)

        if dist_h == 0:
            return 0.0

        mar = (dist_v1 + dist_v2 + dist_v3) / (3.0 * dist_h)
        return mar

    def _calcular_perclos(self):
        """
        Calcula o PERCLOS (Percentage of Eye Closure).

        PERCLOS = (T_fechado / T_total) * 100

        Mede o percentual de tempo em que as palpebras estao 80% ou mais
        fechadas dentro de uma janela temporal (Wierwille et al., 1994).
        """
        if len(self.historico_ear) < 10:
            return 0.0

        frames_fechados = sum(1 for ear in self.historico_ear
                              if ear < config.EAR_LIMIAR)
        total_frames = len(self.historico_ear)

        self.perclos = (frames_fechados / total_frames) * 100
        return self.perclos

    def _calcular_hpe(self, landmarks, w, h):
        """
        Calcula Head Pose Estimation (HPE) usando solvePnP.

        Estima a orientacao da cabeca em tres eixos:
        - Pitch: inclinacao para frente/tras
        - Yaw: rotacao esquerda/direita
        - Roll: inclinacao lateral

        Baseado em Murphy-Chutorian & Trivedi (2009).
        """
        pontos = config.HPE_PONTOS

        # Pontos 2D da imagem
        pontos_2d = np.array([
            self._obter_ponto(landmarks, pontos['nariz'], w, h),
            self._obter_ponto(landmarks, pontos['queixo'], w, h),
            self._obter_ponto(landmarks, pontos['olho_esquerdo'], w, h),
            self._obter_ponto(landmarks, pontos['olho_direito'], w, h),
            self._obter_ponto(landmarks, pontos['boca_esquerda'], w, h),
            self._obter_ponto(landmarks, pontos['boca_direita'], w, h)
        ], dtype=np.float64)

        # Matriz de camera (aproximada)
        focal_length = w
        centro = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, centro[0]],
            [0, focal_length, centro[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Resolver PnP para obter rotacao e translacao
        sucesso, rotation_vec, translation_vec = cv2.solvePnP(
            self.pontos_3d_modelo,
            pontos_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if sucesso:
            # Converter vetor de rotacao para matriz de rotacao
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)

            # Extrair angulos de Euler
            proj_matrix = np.hstack((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                proj_matrix
            )

            self.pitch = euler_angles[0][0]
            self.yaw = euler_angles[1][0]
            self.roll = euler_angles[2][0]

            # Normalizar angulos
            if self.pitch > 180:
                self.pitch -= 360
            if self.yaw > 180:
                self.yaw -= 360
            if self.roll > 180:
                self.roll -= 360

    def _atualizar_estado_olhos(self):
        """Atualiza o estado de deteccao dos olhos com base no EAR."""
        self.frames_processados += 1

        # Durante warm-up: descartar dados instáveis
        if not self.warmup_completo:
            if self.frames_processados >= self.WARMUP_FRAMES:
                self.warmup_completo = True
                self.historico_ear.clear()
                self.frames_olhos_fechados = 0
                self.olhos_fechados = False
            return

        # Registrar no historico para PERCLOS (so apos warm-up)
        self.historico_ear.append(self.ear_medio)

        if self.ear_medio < config.EAR_LIMIAR:
            self.frames_olhos_fechados += 1
            if not self.olhos_fechados:
                self.olhos_fechados = True
                self.olhos_fechados_inicio = time.time()
        else:
            self.frames_olhos_fechados = 0
            self.olhos_fechados = False
            self.olhos_fechados_inicio = 0

    def _atualizar_estado_boca(self):
        """Atualiza o estado de deteccao de bocejo com base no MAR."""
        agora = time.time()

        if self.mar > config.MAR_LIMIAR:
            if not self.boca_aberta:
                self.boca_aberta = True
                self.boca_aberta_inicio = agora

            # Verifica duracao da boca aberta
            if agora - self.boca_aberta_inicio > config.MAR_TEMPO_BOCEJO:
                self.bocejando = True
                # Contagem com debounce
                if agora - self.ultimo_bocejo > config.MAR_DEBOUNCE:
                    self.bocejo_contador += 1
                    self.ultimo_bocejo = agora
        else:
            self.boca_aberta = False
            self.bocejando = False
            # Resetar contador apos periodo sem bocejos
            if agora - self.ultimo_bocejo > config.MAR_RESET_TEMPO:
                self.bocejo_contador = 0

    def _atualizar_estado_cabeca(self):
        """Atualiza o estado de deteccao da posicao da cabeca."""
        agora = time.time()

        # Verifica se a cabeca esta inclinada para baixo (pitch)
        cabeca_inclinada = (
            self.pitch > config.HPE_PITCH_LIMIAR or
            abs(self.roll) > config.HPE_ROLL_LIMIAR
        )

        if cabeca_inclinada:
            if not self.cabeca_baixa:
                self.cabeca_baixa = True
                self.cabeca_baixa_inicio = agora
        else:
            self.cabeca_baixa = False
            self.cabeca_baixa_inicio = 0

    def _classificar_estado(self):
        """
        Classifica o estado do condutor com base em regras claras:

        ALERTA (dispara alarme) quando:
        - 3 ou mais bocejos detectados
        - Olhos fechados por 2 segundos ou mais
        - Cabeca inclinada (frente/tras) por 2 segundos ou mais

        NORMAL em todos os outros casos.
        """
        # Aguardar warm-up para evitar falsos alertas
        if not self.warmup_completo:
            self.estado_atual = config.ESTADO_ALERTA
            return

        agora = time.time()

        # Calcular PERCLOS (para exibicao na interface)
        self._calcular_perclos()

        # --- Regra 1: 3 bocejos → alerta ---
        if self.bocejo_contador >= config.MAR_BOCEJOS_ALERTA:
            self.estado_atual = config.ESTADO_SONOLENCIA_CRITICA
            return

        # --- Regra 2: Olhos fechados por 2 segundos → alerta ---
        if (self.olhos_fechados and self.olhos_fechados_inicio > 0 and
                agora - self.olhos_fechados_inicio >= config.EAR_TEMPO_FECHADO):
            self.estado_atual = config.ESTADO_SONOLENCIA_CRITICA
            return

        # --- Regra 3: Cabeca inclinada por 2 segundos → alerta ---
        if (self.cabeca_baixa and self.cabeca_baixa_inicio > 0 and
                agora - self.cabeca_baixa_inicio >= config.HPE_TEMPO_ALERTA):
            self.estado_atual = config.ESTADO_SONOLENCIA_CRITICA
            return

        # --- Caso contrario: normal ---
        self.estado_atual = config.ESTADO_ALERTA

    def processar_frame(self, img):
        """
        Processa um frame da camera para deteccao de fadiga.

        Retorna:
            img: Imagem original (sem anotacoes visuais)
            metricas: Dicionario com todas as metricas calculadas
            estado: Estado classificado do condutor
        """
        h, w, _ = img.shape

        # Calcular FPS real
        agora = time.time()
        dt = agora - self.ultimo_frame_time
        if dt > 0:
            self.fps_real = 1.0 / dt
        self.ultimo_frame_time = agora

        # Converter para RGB e processar com MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        self.face_detectada = False
        self.ultimo_face_landmarks = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.face_detectada = True
                self.ultimo_face_landmarks = face_landmarks

                # Calcular EAR (ambos os olhos)
                self.ear_direito = self._calcular_ear(
                    face_landmarks, config.OLHO_DIREITO, w, h)
                self.ear_esquerdo = self._calcular_ear(
                    face_landmarks, config.OLHO_ESQUERDO, w, h)
                self.ear_medio = (self.ear_direito + self.ear_esquerdo) / 2.0

                # Calcular MAR
                self.mar = self._calcular_mar(face_landmarks, w, h)

                # Calcular HPE
                self._calcular_hpe(face_landmarks, w, h)

                # Atualizar estados individuais
                self._atualizar_estado_olhos()
                self._atualizar_estado_boca()
                self._atualizar_estado_cabeca()

                # Classificar estado geral
                self._classificar_estado()

                # So processar primeira face
                break

        # Montar dicionario de metricas
        metricas = {
            'ear_esquerdo': self.ear_esquerdo,
            'ear_direito': self.ear_direito,
            'ear_medio': self.ear_medio,
            'mar': self.mar,
            'perclos': self.perclos,
            'pitch': self.pitch,
            'yaw': self.yaw,
            'roll': self.roll,
            'olhos_fechados': self.olhos_fechados,
            'bocejando': self.bocejando,
            'bocejo_contador': self.bocejo_contador,
            'cabeca_baixa': self.cabeca_baixa,
            'face_detectada': self.face_detectada,
            'fps': self.fps_real,
            'frames_olhos_fechados': self.frames_olhos_fechados
        }

        return img, metricas, self.estado_atual

    def obter_landmarks_visuais(self, face_landmarks, w, h):
        """Retorna coordenadas dos landmarks para visualizacao."""
        pontos = {}

        # Olho direito
        for nome, idx in config.OLHO_DIREITO.items():
            pontos[f'od_{nome}'] = self._obter_ponto(face_landmarks, idx, w, h)

        # Olho esquerdo
        for nome, idx in config.OLHO_ESQUERDO.items():
            pontos[f'oe_{nome}'] = self._obter_ponto(face_landmarks, idx, w, h)

        # Boca
        for nome, idx in config.BOCA.items():
            pontos[f'boca_{nome}'] = self._obter_ponto(face_landmarks, idx, w, h)

        # HPE
        for nome, idx in config.HPE_PONTOS.items():
            pontos[f'hpe_{nome}'] = self._obter_ponto(face_landmarks, idx, w, h)

        return pontos

    def liberar(self):
        """Libera recursos do MediaPipe."""
        self.face_mesh.close()
