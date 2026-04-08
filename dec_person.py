import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

   # 3. Executa a detecção no frame atual com uma confiança mínima de 0.5 e filtrando apenas a classe 'pessoa' (classe 0)
    results = model(frame, stream=True, conf=0.5, classes=0)

    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("YOLOv8 Real-Time", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()