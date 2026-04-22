"""
Modulo de Interface Visual (UI Overlay)
========================================
Renderiza uma interface visual profissional sobre o frame da camera,
exibindo metricas em tempo real, estado do condutor e alertas visuais.

Layout:
- Painel superior: Titulo do sistema e FPS
- Painel esquerdo: Metricas (EAR, MAR, PERCLOS, HPE)
- Painel direito: Estado do condutor com indicador visual
- Barras de progresso para cada metrica
- Landmarks faciais com cores por regiao
"""

import cv2
import numpy as np
import time

import detector_fadiga.config as config


class InterfaceVisual:
    """Renderiza a interface visual sobre o frame da camera."""

    def __init__(self):
        self.piscar_estado = True
        self.ultimo_piscar = time.time()
        self.FONTE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONTE_BOLD = cv2.FONT_HERSHEY_DUPLEX

    def _desenhar_retangulo_transparente(self, img, x, y, w, h,
                                         cor=(0, 0, 0), alpha=0.6):
        """Desenha um retangulo semi-transparente."""
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), cor, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _desenhar_barra_progresso(self, img, x, y, largura, valor,
                                   valor_max, cor, label="", mostrar_pct=True):
        """Desenha uma barra de progresso horizontal."""
        altura = 14
        progresso = min(valor / valor_max, 1.0) if valor_max > 0 else 0

        # Fundo da barra
        cv2.rectangle(img, (x, y), (x + largura, y + altura),
                       (50, 50, 50), -1)

        # Barra de progresso
        barra_w = int(largura * progresso)
        if barra_w > 0:
            cv2.rectangle(img, (x, y), (x + barra_w, y + altura), cor, -1)

        # Borda
        cv2.rectangle(img, (x, y), (x + largura, y + altura),
                       (100, 100, 100), 1)

        # Label
        if label:
            cv2.putText(img, label, (x, y - 4),
                        self.FONTE, 0.35, (200, 200, 200), 1)

        # Valor percentual
        if mostrar_pct:
            texto = f"{progresso * 100:.0f}%"
            cv2.putText(img, texto, (x + largura + 5, y + 11),
                        self.FONTE, 0.35, (200, 200, 200), 1)

    def _cor_por_valor(self, valor, limiar_bom, limiar_alerta):
        """Retorna cor baseada em limiar (verde -> amarelo -> vermelho)."""
        if valor <= limiar_bom:
            return (0, 200, 0)      # Verde
        elif valor <= limiar_alerta:
            return (0, 200, 255)    # Amarelo
        else:
            return (0, 0, 255)      # Vermelho

    def _cor_ear(self, ear):
        """Retorna cor para o indicador EAR (invertido: menor = pior)."""
        if ear >= config.EAR_LIMIAR * 1.3:
            return (0, 200, 0)
        elif ear >= config.EAR_LIMIAR:
            return (0, 200, 255)
        else:
            return (0, 0, 255)

    def desenhar_painel_superior(self, img, metricas):
        """Desenha o painel superior com titulo e FPS."""
        h, w = img.shape[:2]

        # Fundo do painel
        self._desenhar_retangulo_transparente(img, 0, 0, w, 45, (30, 30, 30), 0.75)

        # Titulo
        titulo = "SISTEMA DE DETECCAO DE FADIGA - DMS"
        cv2.putText(img, titulo, (15, 30),
                    self.FONTE_BOLD, 0.65, (255, 255, 255), 1)

        # FPS
        fps_texto = f"FPS: {metricas['fps']:.0f}"
        fps_cor = (0, 200, 0) if metricas['fps'] >= 25 else (0, 0, 255)
        cv2.putText(img, fps_texto, (w - 110, 30),
                    self.FONTE, 0.55, fps_cor, 1)

        # Linha separadora
        cv2.line(img, (0, 45), (w, 45), (80, 80, 80), 1)

    def desenhar_painel_metricas(self, img, metricas):
        """Desenha o painel esquerdo com metricas biometricas."""
        # Fundo do painel
        self._desenhar_retangulo_transparente(img, 5, 52, 230, 250, (20, 20, 20), 0.7)

        x_base = 15
        y_base = 68

        # Titulo do painel
        cv2.putText(img, "METRICAS BIOMETRICAS", (x_base, y_base),
                    self.FONTE, 0.42, (150, 200, 255), 1)
        y_base += 8
        cv2.line(img, (x_base, y_base), (x_base + 200, y_base),
                 (60, 60, 60), 1)

        # --- EAR ---
        y_base += 18
        ear = metricas['ear_medio']
        cor_ear = self._cor_ear(ear)
        cv2.putText(img, f"EAR: {ear:.3f}", (x_base, y_base),
                    self.FONTE, 0.42, cor_ear, 1)
        y_base += 5
        self._desenhar_barra_progresso(
            img, x_base, y_base, 180, ear, 0.45, cor_ear,
            mostrar_pct=False)

        # EAR por olho
        y_base += 22
        cv2.putText(img,
                    f"  D: {metricas['ear_direito']:.3f}  "
                    f"E: {metricas['ear_esquerdo']:.3f}",
                    (x_base, y_base), self.FONTE, 0.33, (180, 180, 180), 1)

        # --- MAR ---
        y_base += 22
        mar = metricas['mar']
        cor_mar = self._cor_por_valor(mar, config.MAR_LIMIAR * 0.7,
                                       config.MAR_LIMIAR)
        status_boca = "BOCEJANDO" if metricas['bocejando'] else "Normal"
        cv2.putText(img, f"MAR: {mar:.3f} ({status_boca})",
                    (x_base, y_base), self.FONTE, 0.42, cor_mar, 1)
        y_base += 5
        self._desenhar_barra_progresso(
            img, x_base, y_base, 180, mar, 1.0, cor_mar,
            mostrar_pct=False)

        # Contador de bocejos
        y_base += 22
        bocejos = metricas['bocejo_contador']
        cor_boc = (0, 0, 255) if bocejos >= config.MAR_BOCEJOS_ALERTA \
            else (200, 200, 200)
        cv2.putText(img, f"  Bocejos: {bocejos}/{config.MAR_BOCEJOS_ALERTA}",
                    (x_base, y_base), self.FONTE, 0.33, cor_boc, 1)

        # --- PERCLOS ---
        y_base += 22
        perclos = metricas['perclos']
        cor_perclos = self._cor_por_valor(perclos,
                                          config.PERCLOS_LIMIAR_MODERADO,
                                          config.PERCLOS_LIMIAR_CRITICO)
        cv2.putText(img, f"PERCLOS: {perclos:.1f}%",
                    (x_base, y_base), self.FONTE, 0.42, cor_perclos, 1)
        y_base += 5
        self._desenhar_barra_progresso(
            img, x_base, y_base, 180, perclos, 100, cor_perclos)

        # --- HPE ---
        y_base += 25
        cv2.putText(img, "HPE (Pose da Cabeca):", (x_base, y_base),
                    self.FONTE, 0.38, (150, 200, 255), 1)

        y_base += 18
        pitch = metricas['pitch']
        yaw = metricas['yaw']
        roll = metricas['roll']

        cor_p = self._cor_por_valor(abs(pitch), 10, config.HPE_PITCH_LIMIAR)
        cor_r = self._cor_por_valor(abs(roll), 10, config.HPE_ROLL_LIMIAR)

        cv2.putText(img, f"  Pitch: {pitch:+.1f}", (x_base, y_base),
                    self.FONTE, 0.35, cor_p, 1)
        y_base += 16
        cv2.putText(img, f"  Yaw:   {yaw:+.1f}", (x_base, y_base),
                    self.FONTE, 0.35, (180, 180, 180), 1)
        y_base += 16
        cv2.putText(img, f"  Roll:  {roll:+.1f}", (x_base, y_base),
                    self.FONTE, 0.35, cor_r, 1)

    def desenhar_painel_estado(self, img, estado, confianca):
        """Desenha o painel direito com o estado do condutor."""
        h, w = img.shape[:2]
        painel_w = 250
        painel_x = w - painel_w - 5

        cor_estado = config.CORES_ESTADO.get(estado, (200, 200, 200))
        nome_estado = config.NOMES_ESTADO.get(estado, "DESCONHECIDO")

        # Fundo do painel
        self._desenhar_retangulo_transparente(
            img, painel_x, 52, painel_w, 110, (20, 20, 20), 0.7)

        # Titulo
        cv2.putText(img, "ESTADO DO CONDUTOR", (painel_x + 15, 72),
                    self.FONTE, 0.42, (150, 200, 255), 1)
        cv2.line(img, (painel_x + 15, 78),
                 (painel_x + painel_w - 15, 78), (60, 60, 60), 1)

        # Indicador de estado (circulo colorido)
        centro_x = painel_x + 25
        centro_y = 108
        cv2.circle(img, (centro_x, centro_y), 14, cor_estado, -1)
        cv2.circle(img, (centro_x, centro_y), 14, (255, 255, 255), 1)

        # Piscar para estados criticos
        agora = time.time()
        if agora - self.ultimo_piscar > 0.5:
            self.piscar_estado = not self.piscar_estado
            self.ultimo_piscar = agora

        mostrar = True
        if estado == config.ESTADO_SONOLENCIA_CRITICA:
            mostrar = self.piscar_estado

        if mostrar:
            cv2.putText(img, nome_estado, (centro_x + 22, centro_y + 5),
                        self.FONTE_BOLD, 0.55, cor_estado, 1)

        # Confianca
        cv2.putText(img, f"Confianca: {confianca * 100:.0f}%",
                    (painel_x + 15, 145),
                    self.FONTE, 0.38, (180, 180, 180), 1)

        # Barra de confianca
        self._desenhar_barra_progresso(
            img, painel_x + 15, 150, painel_w - 35,
            confianca, 1.0, cor_estado, mostrar_pct=False)

    def desenhar_alerta_tela_cheia(self, img, estado):
        """Desenha alerta visual em tela cheia para estados criticos."""
        h, w = img.shape[:2]

        if estado == config.ESTADO_SONOLENCIA_CRITICA and self.piscar_estado:
            # Borda vermelha piscante
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
            cv2.rectangle(img, (3, 3), (w - 4, h - 4), (0, 0, 200), 2)

            # Mensagem de alerta central
            self._desenhar_retangulo_transparente(
                img, w // 2 - 220, h - 80, 440, 45, (0, 0, 180), 0.8)

            cv2.putText(img, "!! ALERTA: SONOLENCIA DETECTADA !!",
                        (w // 2 - 200, h - 52),
                        self.FONTE_BOLD, 0.7, (255, 255, 255), 2)

        elif estado == config.ESTADO_FADIGA_MODERADA:
            # Borda amarela
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 180, 255), 2)

            # Mensagem de aviso
            self._desenhar_retangulo_transparente(
                img, w // 2 - 180, h - 70, 360, 35, (0, 140, 200), 0.7)

            cv2.putText(img, "ATENCAO: Sinais de fadiga detectados",
                        (w // 2 - 165, h - 48),
                        self.FONTE, 0.5, (255, 255, 255), 1)

    def desenhar_landmarks(self, img, detector, face_landmarks):
        """Desenha os landmarks faciais com cores por regiao."""
        h, w = img.shape[:2]

        pontos = detector.obter_landmarks_visuais(face_landmarks, w, h)

        # Olho Direito - Verde
        for nome, pt in pontos.items():
            if nome.startswith('od_'):
                cv2.circle(img, pt, 2, (0, 220, 0), -1)

        # Olho Esquerdo - Azul
        for nome, pt in pontos.items():
            if nome.startswith('oe_'):
                cv2.circle(img, pt, 2, (255, 150, 0), -1)

        # Boca - Magenta
        for nome, pt in pontos.items():
            if nome.startswith('boca_'):
                cv2.circle(img, pt, 2, (255, 0, 255), -1)

        # HPE - Ciano
        for nome, pt in pontos.items():
            if nome.startswith('hpe_'):
                cv2.circle(img, pt, 2, (255, 255, 0), -1)

    def desenhar_status_face(self, img, metricas):
        """Desenha indicador de deteccao de face."""
        h, w = img.shape[:2]

        if not metricas['face_detectada']:
            self._desenhar_retangulo_transparente(
                img, w // 2 - 150, h // 2 - 25, 300, 50, (0, 0, 100), 0.8)
            cv2.putText(img, "FACE NAO DETECTADA",
                        (w // 2 - 115, h // 2 + 8),
                        self.FONTE_BOLD, 0.65, (0, 100, 255), 1)

    def desenhar_legenda(self, img):
        """Desenha legenda dos indicadores no canto inferior esquerdo."""
        h, w = img.shape[:2]
        y_base = h - 90

        self._desenhar_retangulo_transparente(
            img, 5, y_base - 5, 195, 88, (20, 20, 20), 0.65)

        items = [
            ("Olho Direito", (0, 220, 0)),
            ("Olho Esquerdo", (255, 150, 0)),
            ("Boca", (255, 0, 255)),
            ("Pose Cabeca", (255, 255, 0))
        ]

        for i, (texto, cor) in enumerate(items):
            y = y_base + 14 + i * 18
            cv2.circle(img, (18, y - 4), 5, cor, -1)
            cv2.putText(img, texto, (30, y),
                        self.FONTE, 0.35, (200, 200, 200), 1)

    def renderizar(self, img, metricas, estado, confianca, detector,
                   face_landmarks=None):
        """
        Renderiza toda a interface visual sobre o frame.

        Parametros:
            img: Frame da camera
            metricas: Dicionario de metricas
            estado: Estado classificado
            confianca: Nivel de confianca da classificacao
            detector: Instancia do DetectorFadiga
            face_landmarks: Landmarks da face (opcional)
        """
        # Desenhar landmarks faciais
        if face_landmarks is not None:
            self.desenhar_landmarks(img, detector, face_landmarks)

        # Paineis de informacao
        self.desenhar_painel_superior(img, metricas)
        self.desenhar_painel_metricas(img, metricas)
        self.desenhar_painel_estado(img, estado, confianca)

        # Legenda
        self.desenhar_legenda(img)

        # Alertas visuais
        self.desenhar_alerta_tela_cheia(img, estado)

        # Status de face
        self.desenhar_status_face(img, metricas)

        return img
