from ultralytics import YOLO
import cv2

# Carregar modelo YOLO (salve10)
model = YOLO("yolov8n.pt")

# Iniciar webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rodar detecção
    results = model(frame)

    # Pegar classes detectadas
    detected = results[0].boxes.cls

    pessoa_detectada = False

    for cls in detected:
        # Classe 0 = pessoa
        if int(cls) == 0:
            pessoa_detectada = True

    # Desenhar resultado
    annotated_frame = results[0].plot()

    # ALERTA
    if pessoa_detectada:
        cv2.putText(annotated_frame, "ALERTA: PESSOA DETECTADA!", 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2)

        print("🚨 Pessoa detectada!")

    # Mostrar tela
    cv2.imshow("Detector de Pessoas", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
