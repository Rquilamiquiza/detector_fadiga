import cv2
import mediapipe as mp
import math
import time

# Inicializa o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Variáveis globais para detecção de sonolência
EYE_CLOSED_THRESHOLD_FRAMES = 90  # 3 segundos * 30 FPS (assumindo 30 FPS)
closed_eye_frame_counter = 0

HEAD_TILT_THRESHOLD_DEGREES = 15 # Limiar de inclinação da cabeça em graus
HEAD_TILT_COUNT_THRESHOLD = 3    # Número de inclinações para disparar alerta
head_tilt_counter = 0
last_head_tilt_time = time.time()
HEAD_TILT_RESET_TIME = 5         # Segundos para resetar o contador de inclinação

TORSO_ANGLE_THRESHOLD = 20       # Limiar de ângulo do tronco (aproximado)

BOCEJO_COUNT_THRESHOLD = 3       # Número de bocejos consecutivos para disparar alerta
bocejo_counter = 0
last_bocejo_time = time.time()

# Variáveis para refinamento da detecção de bocejo
is_mouth_open = False
mouth_open_start_time = 0
TEMPO_BOCA_ABERTA = 0.7 # Segundos que a boca deve permanecer aberta para ser considerado bocejo
ABERTURA_BOCA = 0.4 # Limiar da razão altura/largura para boca aberta

def calcular_distancia(p1, p2):
    """Calcula a distância euclidiana entre dois pontos."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def detectar_olhos_fechados(img, face_landmarks, w, h):
    """Detecta se os olhos estão fechados e atualiza o contador.
    Retorna a imagem com as anotações e o status de alerta, e um booleano indicando sonolência.
    """
    global closed_eye_frame_counter
    drowsiness_alert = False

    # Landmarks para o olho direito (do ponto de vista da câmera)
    # Pontos 159 (superior) e 145 (inferior) para o olho direito
    right_top = face_landmarks.landmark[159]
    right_bottom = face_landmarks.landmark[145]
    # Landmarks para o olho esquerdo (do ponto de vista da câmera)
    # Pontos 386 (superior) e 374 (inferior) para o olho esquerdo
    left_top = face_landmarks.landmark[386]
    left_bottom = face_landmarks.landmark[374]

    # Converte as coordenadas normalizadas para pixels
    rt = int(right_top.x * w), int(right_top.y * h)
    rb = int(right_bottom.x * w), int(right_bottom.y * h)
    lt = int(left_top.x * w), int(left_top.y * h)
    lb = int(left_bottom.x * w), int(left_bottom.y * h)

    # Desenha círculos nas landmarks dos olhos para visualização
    cv2.circle(img, rt, 2, (0, 255, 0), -1)
    cv2.circle(img, rb, 2, (0, 255, 0), -1)
    cv2.circle(img, lt, 2, (255, 0, 0), -1)
    cv2.circle(img, lb, 2, (255, 0, 0), -1)

    # Calcula a distância vertical entre as landmarks superior e inferior de cada olho
    dist_right = calcular_distancia(rt, rb)
    dist_left = calcular_distancia(lt, lb)

    # Determina o status de cada olho (aberto/fechado) com base na distância
    status_right = "Fechado" if dist_right < 5 else "Aberto"
    status_left = "Fechado" if dist_left < 5 else "Aberto"

    # Verifica se ambos os olhos estão fechados
    if status_right == "Fechado" and status_left == "Fechado":
        closed_eye_frame_counter += 1
    else:
        closed_eye_frame_counter = 0

    # Exibe o status dos olhos na imagem
    cv2.putText(img, f"Olho Direito: {status_right}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Olho Esquerdo: {status_left}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Alerta se os olhos estiverem fechados por muito tempo
    if closed_eye_frame_counter >= EYE_CLOSED_THRESHOLD_FRAMES:
        cv2.putText(img, "ALERTA: OLHOS FECHADOS POR MUITO TEMPO!", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        drowsiness_alert = True

    return img, drowsiness_alert

def detectar_inclinacao_cabeca(img, face_landmarks, w, h):
    """Detecta a inclinação da cabeça e atualiza o contador.
    Retorna a imagem com as anotações e o status de alerta, e um booleano indicando sonolência.
    """
    global head_tilt_counter, last_head_tilt_time
    drowsiness_alert = False

    # Landmarks para a inclinação da cabeça (ponta do nariz e orelhas)
    # Ponto 1 (ponta do nariz), 234 (orelha esquerda), 454 (orelha direita)
    nose_tip = face_landmarks.landmark[1]
    left_ear = face_landmarks.landmark[234]
    right_ear = face_landmarks.landmark[454]

    # Converte as coordenadas normalizadas para pixels
    nt = int(nose_tip.x * w), int(nose_tip.y * h)
    le = int(left_ear.x * w), int(left_ear.y * h)
    re = int(right_ear.x * w), int(right_ear.y * h)

    # Desenha círculos nas landmarks da cabeça para visualização
    cv2.circle(img, nt, 2, (255, 255, 0), -1)
    cv2.circle(img, le, 2, (255, 255, 0), -1)
    cv2.circle(img, re, 2, (255, 255, 0), -1)

    # Calcula o ângulo de inclinação da cabeça
    if le[0] != re[0]: # Evita divisão por zero
        angle_rad = math.atan2(le[1] - re[1], le[0] - re[0])
        angle_deg = math.degrees(angle_rad)

        # Normaliza o ângulo para estar entre -180 e 180 graus
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        # Verifica se a cabeça está inclinada além do limiar
        if abs(angle_deg) > HEAD_TILT_THRESHOLD_DEGREES:
            # Evita contagem excessiva no mesmo tilt, espera 1 segundo entre contagens
            if time.time() - last_head_tilt_time > 1:
                head_tilt_counter += 1
                last_head_tilt_time = time.time()
        # Reseta o contador se não houver inclinação por um tempo
        elif time.time() - last_head_tilt_time > HEAD_TILT_RESET_TIME:
            head_tilt_counter = 0

        # Exibe o ângulo de inclinação da cabeça na imagem
        cv2.putText(img, f"Inclinacao Cabeca: {int(angle_deg)} graus", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Alerta se a cabeça inclinou muitas vezes
    if head_tilt_counter >= HEAD_TILT_COUNT_THRESHOLD:
        cv2.putText(img, "ALERTA: CABECA INCLINADA MUITAS VEZES!", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        drowsiness_alert = True

    return img, drowsiness_alert

def detectar_postura_tronco(img, face_landmarks, w, h):
    """Detecta a postura do tronco com base na posição vertical do nariz.
    Retorna a imagem com as anotações e o status de alerta, e um booleano indicando sonolência.
    """
    drowsiness_alert = False

    # Para detectar a postura do tronco, usamos a posição vertical relativa
    # de landmarks da face (como o nariz) em relação ao centro da imagem.
    # Se a pessoa estiver inclinada para frente (dormindo), o nariz estará
    # mais abaixo no frame.

    nose_tip = face_landmarks.landmark[1] # Ponto 1 (ponta do nariz)

    nt = int(nose_tip.x * w), int(nose_tip.y * h)

    # Desenha círculo na landmark do nariz para visualização
    cv2.circle(img, nt, 2, (0, 255, 255), -1)

    # Calcula a posição Y relativa do nariz em relação ao centro da imagem
    center_y = h // 2
    nose_y_relative = nt[1] - center_y
    
    # Se o nariz estiver muito abaixo do centro da imagem, pode indicar sono
    # O limiar de 0.2 (20% da altura da imagem) pode ser ajustado conforme necessário
    if nose_y_relative > h * 0.2:
        cv2.putText(img, "ALERTA: POSTURA INDICA SONO!", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        postura_status = "Sono detectado"
        drowsiness_alert = True
    else:
        postura_status = "Normal"

    # Exibe o status da postura na imagem
    cv2.putText(img, f"Postura: {postura_status}", (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return img, drowsiness_alert

def detectar_bocejo(img, face_landmarks, w, h):
    """Detecta bocejos com base na abertura da boca, duração e velocidade.
    """
    global bocejo_counter, last_bocejo_time, is_mouth_open, mouth_open_start_time
    drowsiness_alert = False
    bocejo_status = "Normal"

    # Landmarks para a boca (superior, inferior, esquerda, direita)
    # Pontos 13 (superior), 14 (inferior), 61 (esquerda), 291 (direita)
    mouth_top = face_landmarks.landmark[13]
    mouth_bottom = face_landmarks.landmark[14]
    mouth_left = face_landmarks.landmark[61]
    mouth_right = face_landmarks.landmark[291]

    # Converte as coordenadas normalizadas para pixels
    mt = int(mouth_top.x * w), int(mouth_top.y * h)
    mb = int(mouth_bottom.x * w), int(mouth_bottom.y * h)
    ml = int(mouth_left.x * w), int(mouth_left.y * h)
    mr = int(mouth_right.x * w), int(mouth_right.y * h)

    # Desenha círculos nas landmarks da boca para visualização
    cv2.circle(img, mt, 2, (255, 0, 255), -1)
    cv2.circle(img, mb, 2, (255, 0, 255), -1)
    cv2.circle(img, ml, 2, (255, 0, 255), -1)
    cv2.circle(img, mr, 2, (255, 0, 255), -1)

    # Calcula a abertura vertical e horizontal da boca
    mouth_height = calcular_distancia(mt, mb)
    mouth_width = calcular_distancia(ml, mr)

    # Calcula a razão altura/largura para detectar bocejo
    if mouth_width > 0:
        mouth_ratio = mouth_height / mouth_width
        
        # Verifica se a boca está aberta o suficiente para um bocejo
        if mouth_ratio > ABERTURA_BOCA:
            if not is_mouth_open:
                is_mouth_open = True
                mouth_open_start_time = time.time()
            
            # Verifica a duração da boca aberta
            if time.time() - mouth_open_start_time > TEMPO_BOCA_ABERTA:
                bocejo_status = "Bocejando"
                # Evita contagem excessiva, espera 2 segundos entre bocejos
                if time.time() - last_bocejo_time > 2:
                    bocejo_counter += 1
                    last_bocejo_time = time.time()
        else:
            bocejo_status = "Normal"
            is_mouth_open = False
            # Reseta o contador se não houver bocejo por um tempo (10 segundos)
            if time.time() - last_bocejo_time > 10:
                bocejo_counter = 0

        # Exibe o status da boca na imagem
        cv2.putText(img, f"Boca: {bocejo_status}", (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Alerta se houve muitos bocejos consecutivos
        if bocejo_counter >= BOCEJO_COUNT_THRESHOLD:
            cv2.putText(img, "ALERTA: MUITOS BOCEJOS CONSECUTIVOS!", (30, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            drowsiness_alert = True

    return img, drowsiness_alert

def processar_frame(img):
    """Processa o frame da câmera para detecção de sonolência.
    Chama as funções de detecção e retorna a imagem com todas as anotações e um status geral de sonolência.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    drowsiness_detected_in_frame = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = img.shape
            
            img, alert_eyes = detectar_olhos_fechados(img, face_landmarks, w, h)
            img, alert_head = detectar_inclinacao_cabeca(img, face_landmarks, w, h)
            img, alert_torso = detectar_postura_tronco(img, face_landmarks, w, h)
            img, alert_yawn = detectar_bocejo(img, face_landmarks, w, h)

            if alert_eyes or alert_head or alert_torso or alert_yawn:
                drowsiness_detected_in_frame = True

    return img, drowsiness_detected_in_frame

