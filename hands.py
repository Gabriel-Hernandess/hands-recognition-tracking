# Código desenvolvido por Gabriel Yanagawa Hernandes, para fins de estudo em IA e Machine Learning

# O código faz com que ao pressionar a tecla b, com a palma da mao para cima, um 'Jogo' comece, seu objetivo e nao deixar cair a bolinha, fazendo o 
# máximo de pontos possíveis. Para funcionar no seu sistema, pode ser necessário algumas alterações. 

# MacOS Ventura 13.3.1

# LinkedIn: https://encurtador.com.br/jmyHK
# GitHub: https://github.com/Gabriel-Hernandess


#libs necessárias
import cv2
import mediapipe as mp
import math
import random

# indicar a webcam / camera a ser usada
cap = cv2.VideoCapture(0)

# iniciar configs Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

ball_falling = False  # Variável de controle para indicar se a bolinha deve começar a cair
ball_position = [0, 0]  # Posição inicial da bolinha
hand_position = [0, 0]  # Posição da palma da mão no frame anterior
throwing = False  # Variável de controle para indicar se a bolinha está sendo jogada para cima
pontos = 0 # pontos da 'Rodada'
best = 0 # Melhor pontuação

while True:
    # ler a webcam / camera
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    cx, cy = 0, 0  # Defina cx e cy com valores padrão
    handLms = None  # Defina handLms com valor padrão

    # verificar se alguma mão foi identificada
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # verificando as maos detectadas (Testado apenas com 1)
            # calcular os pontos 5 e 9
            angle = math.degrees(math.atan2(handLms.landmark[9].y - handLms.landmark[5].y,
                                           handLms.landmark[9].x - handLms.landmark[5].x))

            # verificar angulo, se a palma estiver pra cima, gera um circulo roxo
            if 0 <= angle <= 90:
                cx, cy = int(handLms.landmark[9].x * image.shape[1]), int(handLms.landmark[9].y * image.shape[0])
                cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

                # verificar se a bolinha tocou na palma da mão
                if not throwing and abs(cx - ball_position[0]) < 25 and abs(cy - ball_position[1]) < 25:
                    throwing = True  # indica que agora, ela ira ser lancada para cima
                    ball_falling = False  # ela para de cair
                    pontos += 1 # aumenta 1 ponto
                    ball_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Cor aleatória

    # Verifique se a tecla 'b' foi pressionada para iniciar a queda da bolinha
    if cv2.waitKey(1) & 0xFF == ord('b') and not ball_falling and not throwing:
        ball_falling = True
        ball_position = [random.randint(0, image.shape[1]), 0]
        ball_color = (0, 255, 0)

    # Atualizar a posição da bolinha se ela estiver caindo
    if ball_falling:

        ball_position[1] += 20  # Ajuste a velocidade de queda conforme necessário

        # Verificar se a bolinha passou do limite inferior (Fail)
        if ball_position[1] > image.shape[0]: 
            ball_falling = False
            throwing = False
            best = pontos
            pontos = 0

        # se nao, continuar desenhando a bolinha 
        else:
            cv2.circle(image, (ball_position[0], ball_position[1]), 25, ball_color, cv2.FILLED)


    # Se a bolinha está sendo jogada para cima, atualize sua posição
    if throwing:
        ball_position[1] -= 30  # Ajuste a velocidade do lançamento conforme necessário

        # Verificar se a bolinha atingiu o ponto mais alto e começar a cair novamente
        if ball_position[1] < image.shape[0]:
            throwing = False
            ball_falling = True
            ball_position = [random.randint(0, image.shape[1]), 0]
        
        # se não, continuar desenhando a bolinha
        else:
            cv2.circle(image, (ball_position[0], ball_position[1]), 25, ball_color, cv2.FILLED)

    # Salvar a posição da palma da mão para o próximo frame
    hand_position = [cx, cy]

    if handLms is not None:
        mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    # Exibir a contagem de toques
    cv2.putText(image, f'Toques: {pontos}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Exibir melhor pontuação
    cv2.putText(image, f'Best Score: {best}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Output", image)

    # Verificar se a tecla 'q' foi pressionada para encerrar o programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()