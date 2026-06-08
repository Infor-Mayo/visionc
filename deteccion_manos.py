import cv2
import mediapipe as mp
import numpy as np
import random

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections
drawing_utils = mp.tasks.vision.drawing_utils
RunningMode = mp.tasks.vision.RunningMode

resultado_manos = None
resultado_cara = None

def hand_callback(result: HandLandmarkerResult, output_image, timestamp_ms: int):
    global resultado_manos
    resultado_manos = result

def face_callback(result: FaceLandmarkerResult, output_image, timestamp_ms: int):
    global resultado_cara
    resultado_cara = result

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    result_callback=hand_callback
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=RunningMode.LIVE_STREAM,
    num_faces=2,
    output_face_blendshapes=True,
    result_callback=face_callback
)

hand_landmarker = HandLandmarker.create_from_options(hand_options)
face_landmarker = FaceLandmarker.create_from_options(face_options)

# --- YOLO ---
with open("coco.names", "r") as f:
    coco_classes = [line.strip() for line in f.readlines()]

colors = {}
for cls in coco_classes:
    colors[cls] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

yolo_net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers().flatten()]

CONF_THRESH = 0.4
NMS_THRESH = 0.5
frame_count = 0
DETECT_EVERY = 3
yolo_detections = []

FINGER_INFO = [
    ("THUMB", HandLandmarksConnections.HAND_THUMB_CONNECTIONS, 4, None),
    ("INDEX", HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS, 8, 6),
    ("MIDDLE", HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS, 12, 10),
    ("RING", HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS, 16, 14),
    ("PINKY", HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS, 20, 18),
]

def is_finger_up(hand_landmarks, finger):
    name, _, tip, pip = finger
    if name == "THUMB":
        return hand_landmarks[tip].x < hand_landmarks[tip - 1].x
    return hand_landmarks[tip].y < hand_landmarks[pip].y

def detect_objects(image):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (320, 320), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)
    h, w = image.shape[:2]
    boxes, confs, class_ids = [], [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESH:
                cx, cy, bw, bh = det[:4] * np.array([w, h, w, h])
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confs.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            cls = coco_classes[class_ids[i]]
            results.append({
                "class": cls,
                "conf": confs[i],
                "box": (x, y, x + bw, y + bh),
                "center": (x + bw // 2, y + bh // 2)
            })
    return results

cap = cv2.VideoCapture(0)
print("Presiona 'q' para salir.")
print("Dedos activos en VERDE, dedos doblados en ROJO. YOLO activo.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("No se pudo acceder a la cámara.")
        break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    hand_landmarker.detect_async(mp_image, timestamp)
    face_landmarker.detect_async(mp_image, timestamp)

    frame_count += 1
    if frame_count % DETECT_EVERY == 0:
        yolo_detections = detect_objects(image)

    for det in yolo_detections:
        x1, y1, x2, y2 = det["box"]
        color = colors[det["class"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} ({det['conf']:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # --- FACE MESH ---
    if resultado_cara and resultado_cara.face_landmarks:
        for face_landmarks in resultado_cara.face_landmarks:
            conexiones = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
            puntos = []
            for lm in face_landmarks:
                puntos.append((int(lm.x * w), int(lm.y * h)))

            for conn in conexiones:
                x1, y1 = puntos[conn.start]
                x2, y2 = puntos[conn.end]
                cv2.line(image, (x1, y1), (x2, y2), (200, 200, 200), 1)

            for px, py in puntos:
                cv2.circle(image, (px, py), 1, (255, 255, 255), -1)

        if resultado_cara.face_blendshapes:
            shapes = resultado_cara.face_blendshapes[0]
            expr_map = {}
            for cat in shapes:
                expr_map[cat.category_name] = cat.score

            expressions = []
            if expr_map.get("mouthSmileLeft", 0) + expr_map.get("mouthSmileRight", 0) > 0.3:
                expressions.append("SONRIENTE")
            if expr_map.get("jawOpen", 0) > 0.5:
                expressions.append("SORPRENDIDO")
            if expr_map.get("browDownLeft", 0) + expr_map.get("browDownRight", 0) > 0.4:
                expressions.append("ENOJADO")
            if expr_map.get("eyeBlinkLeft", 0) > 0.5 and expr_map.get("eyeBlinkRight", 0) > 0.5:
                expressions.append("OJOS CERRADOS")

            if expressions:
                cv2.putText(image, f'Expresion: {", ".join(expressions)}', (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # --- HAND SKELETON ---
    if resultado_manos and resultado_manos.hand_landmarks:
        for idx, hand_landmarks in enumerate(resultado_manos.hand_landmarks):
            if idx >= len(resultado_manos.handedness):
                continue
            fingers_up = []
            for finger in FINGER_INFO:
                fingers_up.append(is_finger_up(hand_landmarks, finger))

            for i, finger in enumerate(FINGER_INFO):
                _, connections, _, _ = finger
                color = (0, 255, 0) if fingers_up[i] else (0, 0, 255)
                for conn in connections:
                    start = hand_landmarks[conn.start]
                    end = hand_landmarks[conn.end]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(image, (x1, y1), (x2, y2), color, 3)
                    cv2.circle(image, (x1, y1), 5, color, -1)
                    cv2.circle(image, (x2, y2), 5, color, -1)

            palm_connections = HandLandmarksConnections.HAND_PALM_CONNECTIONS
            for conn in palm_connections:
                start = hand_landmarks[conn.start]
                end = hand_landmarks[conn.end]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            total_fingers = fingers_up.count(1)
            cx = int(np.mean([lm.x for lm in hand_landmarks]) * w)
            cy = int(np.mean([lm.y for lm in hand_landmarks]) * h)
            label = "Izquierda" if resultado_manos.handedness[idx][0].category_name == "Left" else "Derecha"

            objeto_cerca = ""
            for det in yolo_detections:
                x1b, y1b, x2b, y2b = det["box"]
                if x1b <= cx <= x2b and y1b <= cy <= y2b:
                    objeto_cerca = det["class"]
                    break

            texto = f'{label}: {total_fingers}'
            if objeto_cerca:
                texto += f' [{objeto_cerca}]'
            cv2.putText(image, texto, (cx - 60, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    cv2.imshow('Deteccion de Manos, Gestos y Expresiones + YOLO', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
face_landmarker.close()
