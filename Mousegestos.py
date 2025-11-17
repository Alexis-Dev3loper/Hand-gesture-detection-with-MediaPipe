import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Resolución de la pantalla
screen_width, screen_height = pyautogui.size()

# pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)

# --- Parámetros de gestos ---
LEFT_PINCH_THRESHOLD = 40
RIGHT_PINCH_THRESHOLD = 40
RIGHT_CLICK_COOLDOWN_FRAMES = 15

is_left_down = False
right_click_cooldown = 0

window_name = "Cursor con la mano"

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Procesar con MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if right_click_cooldown > 0:
            right_click_cooldown -= 1

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Coordenadas del índice (landmark 8)
                x_index = int(hand_landmarks.landmark[8].x * w)
                y_index = int(hand_landmarks.landmark[8].y * h)

                # Mover ratón con el índice
                screen_x = int(hand_landmarks.landmark[8].x * screen_width)
                screen_y = int(hand_landmarks.landmark[8].y * screen_height)
                pyautogui.moveTo(screen_x, screen_y)

                # Coordenadas del pulgar (landmark 4)
                x_thumb = int(hand_landmarks.landmark[4].x * w)
                y_thumb = int(hand_landmarks.landmark[4].y * h)

                # Coordenadas del dedo medio (landmark 12)
                x_middle = int(hand_landmarks.landmark[12].x * w)
                y_middle = int(hand_landmarks.landmark[12].y * h)

                # Dibujar puntos de referencia
                cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), -1)
                cv2.circle(frame, (x_thumb, y_thumb), 10, (0, 0, 255), -1)
                cv2.circle(frame, (x_middle, y_middle), 10, (255, 0, 0), -1)

                # Distancias entre dedos
                dist_thumb_index = ((x_index - x_thumb) ** 2 +
                                    (y_index - y_thumb) ** 2) ** 0.5
                dist_thumb_middle = ((x_middle - x_thumb) ** 2 +
                                     (y_middle - y_thumb) ** 2) ** 0.5

                # ---------- CLICK DERECHO (pulgar + índice + medio) ----------
                triple_pinch = (dist_thumb_index < RIGHT_PINCH_THRESHOLD and
                                dist_thumb_middle < RIGHT_PINCH_THRESHOLD)

                if triple_pinch and right_click_cooldown == 0:
                    # Por si estuviera arrastrando con botón izq
                    if is_left_down:
                        pyautogui.mouseUp(button="left")
                        is_left_down = False

                    pyautogui.click(button="right")
                    right_click_cooldown = RIGHT_CLICK_COOLDOWN_FRAMES

                # ---------- CLICK / DRAG IZQUIERDO (pulgar + índice) ----------
                # Solo si NO estamos en triple pinch (así no se mezcla con el derecho)
                if not triple_pinch:
                    if dist_thumb_index < LEFT_PINCH_THRESHOLD:
                        # Mantener pulgar-índice juntos: mouseDown (permite arrastrar)
                        if not is_left_down:
                            pyautogui.mouseDown(button="left")
                            is_left_down = True
                    else:
                        # Separar los dedos: mouseUp
                        if is_left_down:
                            pyautogui.mouseUp(button="left")
                            is_left_down = False

        else:
            # Si se pierde la mano y el botón estaba pulsado, lo soltamos por seguridad
            if is_left_down:
                pyautogui.mouseUp(button="left")
                is_left_down = False

        cv2.imshow(window_name, frame)

        # Gestión de teclado y cierre de ventana
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC o 'q'
            break

        # Si el usuario cierra la ventana con la X
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
