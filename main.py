"""
Sistema de Deteccao de Fadiga e Sono em Condutores de Veiculos
================================================================
Utilizando Visao Computacional e Aprendizado de Maquina

Trabalho de Fim de Curso - Universidade Oscar Ribas
Autor: Romao Henrique Albano Quilamiquiza
Orientador: MSc. Helder Chissingui

Arquitetura do Sistema (conforme Figura 3.2 do TCC):
    Camera RGB -> Pre-processamento (OpenCV/MediaPipe)
    -> Extracao de Atributos (EAR, MAR, HPE)
    -> Modelo de Classificacao (CNN / Logica de Limiar)
    -> Modulo de Decisao -> Interface de Alerta

Controles:
    ESC   - Encerrar o sistema
    S     - Silenciar/Parar alerta sonoro
    R     - Reiniciar contadores
"""

import cv2
import sys
import os

# Adicionar diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detector_fadiga.config as config
from detector_fadiga.src.detector import DetectorFadiga
from detector_fadiga.src.classificador import ClassificadorFadiga
from detector_fadiga.src.alerta import SistemaAlerta
from detector_fadiga.src.ui import InterfaceVisual


def main():
    """Funcao principal do sistema de deteccao de fadiga."""

    print("=" * 60)
    print("  SISTEMA DE DETECCAO DE FADIGA E SONOLENCIA")
    print("  Driver Monitoring System (DMS)")
    print("=" * 60)
    print()
    print("  Inicializando componentes...")

    # Inicializar modulos
    detector = DetectorFadiga()
    classificador = ClassificadorFadiga()
    alerta = SistemaAlerta()
    ui = InterfaceVisual()

    print("  [OK] Detector de fadiga inicializado")
    print("  [OK] Classificador inicializado")
    print("  [OK] Sistema de alerta inicializado")
    print("  [OK] Interface visual inicializada")

    # Configurar camera
    print(f"\n  Abrindo camera (ID: {config.CAMERA_ID})...")
    cap = cv2.VideoCapture(config.CAMERA_ID)

    if not cap.isOpened():
        print("  [ERRO] Nao foi possivel abrir a camera.")
        print("  Verifique se a webcam esta conectada e disponivel.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_LARGURA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_ALTURA)

    largura_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  [OK] Camera aberta: {largura_real}x{altura_real}")

    print("\n  Sistema pronto!")
    print("  Controles: ESC=Sair | S=Silenciar | R=Reiniciar")
    print("-" * 60)

    # Nome da janela
    nome_janela = "Sistema de Deteccao de Fadiga - DMS"

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("  [ERRO] Falha na captura do frame.")
                break

            # Espelhar frame (mais natural para o usuario)
            frame = cv2.flip(frame, 1)

            # 1. Processar frame (extracao de metricas)
            frame_processado, metricas, estado_detector = \
                detector.processar_frame(frame)

            # 2. Classificar estado do condutor
            # Usar classificacao do detector (baseada em regras dos indicadores)
            estado = estado_detector

            # Obter confianca do modelo quando disponivel
            if metricas['face_detectada'] and detector.warmup_completo:
                _, confianca = classificador.classificar(metricas)
            else:
                confianca = 0.0

            # 3. Atualizar sistema de alerta
            alerta.atualizar(estado)

            # 4. Obter landmarks para visualizacao (ja processados)
            face_landmarks = detector.ultimo_face_landmarks

            # 5. Renderizar interface visual
            frame_final = ui.renderizar(
                frame_processado, metricas, estado, confianca,
                detector, face_landmarks
            )

            # Exibir frame
            cv2.imshow(nome_janela, frame_final)

            # Processar teclas
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC - Sair
                print("\n  Sistema encerrado pelo usuario.")
                break

            elif key == ord('s') or key == ord('S'):  # Silenciar
                alerta.parar()
                print("  [INFO] Alerta silenciado.")

            elif key == ord('r') or key == ord('R'):  # Reiniciar
                detector.frames_olhos_fechados = 0
                detector.bocejo_contador = 0
                detector.perclos = 0.0
                detector.historico_ear.clear()
                alerta.parar()
                print("  [INFO] Contadores reiniciados.")

    except KeyboardInterrupt:
        print("\n  Sistema interrompido.")

    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        alerta.liberar()
        detector.liberar()
        print("  Recursos liberados. Ate logo!")


if __name__ == "__main__":
    main()
