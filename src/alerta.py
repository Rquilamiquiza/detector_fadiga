"""
Modulo de Sistema de Alerta
============================
Gerencia alertas sonoros e visuais conforme o estado de fadiga
detectado no condutor.

Niveis de alerta:
- Fadiga Moderada: alerta visual preventivo
- Sonolencia Critica: alerta sonoro de alta intensidade

Conforme descrito na secao 3.3 do TCC.
"""

import threading
import time
import os

import detector_fadiga.config as config


class SistemaAlerta:
    """Gerencia alertas sonoros e visuais para o condutor."""

    def __init__(self):
        self.alertando = False
        self.thread_alerta = None
        self.estado_alerta = config.ESTADO_ALERTA
        self.som_disponivel = False
        self.beep_disponivel = False

        # Inicializar pygame para audio
        try:
            import pygame
            pygame.mixer.init()
            caminho_som = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "alert.mp3"
            )
            if os.path.exists(caminho_som):
                pygame.mixer.music.load(caminho_som)
                self.som_disponivel = True
            self.pygame = pygame
        except Exception:
            self.pygame = None

        # Verificar disponibilidade do winsound
        try:
            import winsound
            self.winsound = winsound
            self.beep_disponivel = True
        except ImportError:
            self.winsound = None

    def _loop_alerta(self):
        """Loop executado na thread de alerta."""
        while self.alertando:
            if self.estado_alerta == config.ESTADO_SONOLENCIA_CRITICA:
                # Alerta sonoro intenso
                if self.som_disponivel and self.pygame:
                    try:
                        if not self.pygame.mixer.music.get_busy():
                            self.pygame.mixer.music.play()
                    except Exception:
                        pass

                if self.beep_disponivel and self.winsound:
                    try:
                        self.winsound.Beep(
                            config.ALERTA_BEEP_FREQ,
                            config.ALERTA_BEEP_DURACAO
                        )
                    except Exception:
                        pass

            elif self.estado_alerta == config.ESTADO_FADIGA_MODERADA:
                # Beep suave de aviso
                if self.beep_disponivel and self.winsound:
                    try:
                        self.winsound.Beep(800, 300)
                    except Exception:
                        pass

            time.sleep(config.ALERTA_INTERVALO)

    def atualizar(self, estado):
        """Atualiza o sistema de alerta com base no estado detectado."""
        estado_anterior = self.estado_alerta
        self.estado_alerta = estado

        if estado == config.ESTADO_ALERTA:
            self.parar()
        elif estado in (config.ESTADO_FADIGA_MODERADA,
                        config.ESTADO_SONOLENCIA_CRITICA):
            if not self.alertando:
                self.iniciar()

    def iniciar(self):
        """Inicia o sistema de alerta em thread separada."""
        if not self.alertando:
            self.alertando = True
            self.thread_alerta = threading.Thread(
                target=self._loop_alerta, daemon=True)
            self.thread_alerta.start()

    def parar(self):
        """Para o sistema de alerta."""
        self.alertando = False
        if self.som_disponivel and self.pygame:
            try:
                self.pygame.mixer.music.stop()
            except Exception:
                pass

    def liberar(self):
        """Libera recursos do sistema de alerta."""
        self.parar()
        if self.pygame:
            try:
                self.pygame.mixer.quit()
            except Exception:
                pass
