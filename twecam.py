import cv2

# 0 = webcam padrão (pode mudar para 1, 2 se tiver mais câmeras)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # captura frame da webcam
    
    if not ret:
        print("Erro ao acessar a câmera")
        break

    cv2.imshow('Webcam', frame)  # mostra o vídeo

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha tudo
cap.release()
cv2.destroyAllWindows()
