#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args


def count_fingers(landmark_list, handedness_label):
    if len(landmark_list) == 0:
        return 0

    thumb_tip = 4
    thumb_ip = 3
    finger_tips = [8, 12, 16, 20]   # índice, medio, anular, meñique
    finger_pips = [6, 10, 14, 18]   # PIP de esos dedos

    fingers_up = 0

    # --- Pulgar ---
    # Imagen está espejada (cv.flip), así que usamos la mano para decidir lado.
    # Aquí solo queremos contar, no usar esto para Like/Dislike.
    if handedness_label == "Right":
        # Pulgar de la derecha apunta hacia la izquierda de la imagen
        if landmark_list[thumb_tip][0] < landmark_list[thumb_ip][0]:
            fingers_up += 1
    else:  # "Left"
        # Pulgar de la izquierda apunta hacia la derecha de la imagen
        if landmark_list[thumb_tip][0] > landmark_list[thumb_ip][0]:
            fingers_up += 1

    # --- 4 dedos restantes ---
    # Dedo levantado si la punta está por encima (y menor) que la PIP
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        if landmark_list[tip_idx][1] < landmark_list[pip_idx][1]:
            fingers_up += 1

    return fingers_up


def _hand_size(landmark_list):
    """
    Tamaño aproximado de la mano: distancia muñeca (0) a base del dedo medio (9).
    Lo usamos para tener umbrales proporcionales al tamaño de la mano.
    """
    palm_base = np.array(landmark_list[0], dtype=np.float32)
    middle_mcp = np.array(landmark_list[9], dtype=np.float32)
    dist = np.linalg.norm(palm_base - middle_mcp)
    return max(dist, 1.0)


def _other_fingers_mostly_closed(landmark_list, tol_ratio: float = 0.15):
    """
    Devuelve True si al menos 3 de los 4 dedos (índice, medio, anular, meñique)
    NO están claramente extendidos (es decir, están doblados o casi pegados a la palma).

    Miramos la diferencia en Y entre la punta y la articulación MCP de cada dedo.
    Si la punta no está MUCHO más arriba que la MCP, lo consideramos "cerrado".
    """
    size = _hand_size(landmark_list)
    tol = tol_ratio * size

    fingers_closed = 0
    for tip_idx, mcp_idx in ((8, 5), (12, 9), (16, 13), (20, 17)):
        tip_y = landmark_list[tip_idx][1]
        mcp_y = landmark_list[mcp_idx][1]

        # Si la punta NO está significativamente por encima de la MCP,
        # lo consideramos "cerrado" (doblado o horizontal).
        if tip_y > mcp_y - tol:
            fingers_closed += 1

    return fingers_closed >= 3  # al menos 3 de 4 cerrados


def detect_gestures(landmark_list, handedness_label):
    if len(landmark_list) == 0:
        return "None"

    # El orden importa
    if is_like_gesture(landmark_list):
        return "Like"
    if is_dislike_gesture(landmark_list):
        return "Dislike"
    if is_ok_gesture(landmark_list):
        return "OK"
    if is_peace_gesture(landmark_list):
        return "Peace"

    return "None"


def is_like_gesture(landmark_list):
    """
    Like: puño (otros dedos cerrados) + pulgar claramente hacia ARRIBA.
    """
    if not _other_fingers_mostly_closed(landmark_list):
        return False

    size = _hand_size(landmark_list)
    thumb_tip_y = landmark_list[4][1]
    thumb_mcp_y = landmark_list[2][1]

    # Diferencia vertical: positiva si la MCP está más abajo que la punta,
    # negativa si la punta está más arriba.
    dy = thumb_mcp_y - thumb_tip_y  # >0 si la punta está más ARRIBA

    # Pedimos que la punta esté al menos ~25% del tamaño de la mano por encima de la MCP
    return dy > 0.25 * size


def is_dislike_gesture(landmark_list):
    """
    Dislike: puño (otros dedos cerrados) + pulgar claramente hacia ABAJO.
    """
    if not _other_fingers_mostly_closed(landmark_list):
        return False

    size = _hand_size(landmark_list)
    thumb_tip_y = landmark_list[4][1]
    thumb_mcp_y = landmark_list[2][1]

    dy = thumb_tip_y - thumb_mcp_y  # >0 si la punta está más ABAJO

    return dy > 0.25 * size


def is_ok_gesture(landmark_list):
    thumb_tip = landmark_list[4]
    index_tip = landmark_list[8]

    # Distancia entre pulgar e índice
    distance = np.sqrt(
        (thumb_tip[0] - index_tip[0]) ** 2 +
        (thumb_tip[1] - index_tip[1]) ** 2
    )

    # Distancia de referencia: muñeca a base del dedo medio
    palm_base = landmark_list[0]
    middle_mcp = landmark_list[9]
    ref_dist = np.sqrt(
        (palm_base[0] - middle_mcp[0]) ** 2 +
        (palm_base[1] - middle_mcp[1]) ** 2
    )

    if ref_dist == 0:
        return False

    # Consideramos "círculo" si están bastante cerca (relativo al tamaño de la mano)
    forming_circle = distance < 0.5 * ref_dist

    # Otros dedos extendidos
    other_fingers_extended = True
    for tip_idx, mcp_idx in zip([12, 16, 20], [9, 13, 17]):
        if landmark_list[tip_idx][1] > landmark_list[mcp_idx][1]:
            other_fingers_extended = False
            break

    return forming_circle and other_fingers_extended


def is_peace_gesture(landmark_list):
    # Índice y medio arriba
    index_up = landmark_list[8][1] < landmark_list[6][1]
    middle_up = landmark_list[12][1] < landmark_list[10][1]

    # Anular y meñique abajo
    ring_down = landmark_list[16][1] > landmark_list[14][1]
    pinky_down = landmark_list[20][1] > landmark_list[18][1]

    return index_up and middle_up and ring_down and pinky_down


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Cargar modelo de detección facial
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # 0 para corto alcance, 1 para largo alcance
        min_detection_confidence=0.5
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0
    number = -1
    window_name = 'Hand Gesture and Face Recognition'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    while True:
        fps = cvFpsCalc.get()

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False

        face_results = face_detection.process(rgb_image)
        results = hands.process(rgb_image)

        rgb_image.flags.writeable = True

        # =================== DETECCIÓN DE CARA ===================
        if face_results and face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = debug_image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                cv.rectangle(debug_image, (x, y), (x + width, y + height),
                             (0, 255, 0), 2)
                cv.putText(debug_image, "Face", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # =================== MANOS + GESTOS ======================
        finger_count_total = 0
        gestures_summary = []  # ["Right:Like", "Left:Dislike", ...]

        if results and results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Label de la mano ("Left" / "Right")
                hand_label = handedness.classification[0].label

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification (modelo original)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification (modelo original)
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Contar dedos por mano y acumular total
                fingers_this_hand = count_fingers(landmark_list, hand_label)
                finger_count_total += fingers_this_hand

                # Detectar gesto por mano (Like, Dislike, OK, Peace, None)
                this_gesture = detect_gestures(landmark_list, hand_label)
                gestures_summary.append(f"{hand_label}:{this_gesture}")

                # Dibujar mano y textos
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks_with_numbers(debug_image,
                                                          landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                # Gesto de esta mano justo encima del rectángulo
                if this_gesture != "None":
                    gesture_text = f"{hand_label}: {this_gesture}"
                    cv.putText(debug_image, gesture_text,
                               (brect[0], brect[1] - 30),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 0, 0), 4, cv.LINE_AA)
                    cv.putText(debug_image, gesture_text,
                               (brect[0], brect[1] - 30),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6,
                               (255, 255, 255), 2, cv.LINE_AA)
        else:
            point_history.append([0, 0])

        # Dibujar historial y HUD /////////////////////////////////////////////////
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Dedos totales (suma de ambas manos)
        cv.putText(debug_image, f"Dedos: {finger_count_total}", (10, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(debug_image, f"Dedos: {finger_count_total}", (10, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        # Resumen de gestos por mano
        if gestures_summary:
            gestos_text = "Gestos: " + " | ".join(gestures_summary)
        else:
            gestos_text = "Gestos: None"

        cv.putText(debug_image, gestos_text, (10, 170),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(debug_image, gestos_text, (10, 170),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        # Screen reflection #############################################################
        cv.imshow(window_name, debug_image)

        # === Gestión de teclado y cierre de ventana ===
        key = cv.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'):   # ESC o 'q'
            break

        # Actualizar modo / número para el logger
        number, mode = select_mode(key, mode)

        # Si el usuario cierra la ventana con la X
        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv.destroyAllWindows()



def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks_with_numbers(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Pinky
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        cv.circle(image, (landmark[0], landmark[1]), 5,
                  (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), 5,
                  (0, 0, 0), 1)

        cv.putText(image, str(index), (landmark[0] + 10, landmark[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
