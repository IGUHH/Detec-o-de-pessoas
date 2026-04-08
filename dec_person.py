import cv2
from ultralytics import YOLO

# 1. Carrega o modelo (o 'n' indica a versão Nano, ideal para tempo real)
model = YOLO('yolov8n.pt')

# 2. Abre a conexão com a webcam (0 costuma ser a câmera integrada)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Executa a detecção no frame atual
    results = model(frame, stream=True)

    # 4. Exibe os resultados no frame
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("YOLOv8 Real-Time", annotated_frame)

    # Aperte 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()