# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0,cv.CAP_DSHOW)

list_char = list(range(33,126))

global grava
grava=False

imagem_fundo = np.full((480,720,3),0,np.uint8)


def retornaCaractere():
    index = np.random.randint(0,92)
    
    return chr(list_char[index])



def inicializaFilme():
    global filme
    filme = cv.VideoWriter('./arte12.mp4',cv.VideoWriter.fourcc(*'mp4v'),15,(1440,960))


while True:
    global filme
    
    isCap,frame =  cap.read()
    
    
    
    if isCap:
        imagemMatrix = imagem_fundo.copy()

        w , h  = imagemMatrix.shape[:2]

        frame = cv.resize(frame,(h ,w))
        
        
        imagem_cinza = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        imagem_cinza = cv.equalizeHist(imagem_cinza)

        imagem_canny = cv.Canny(imagem_cinza,100,100) 
        
        
        invertida = cv.bitwise_not(imagem_canny)
        
        contornos, _ = cv.findContours(invertida,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        
        
        for contorno in contornos:
            
            x,y = contorno[0][0]                       

            
            cv.putText(imagemMatrix,retornaCaractere(),(x,y),cv.FONT_HERSHEY_COMPLEX_SMALL,0.3,(32,255,66),1,cv.LINE_AA)
            
        
        cv.putText(frame,'(q - sair) (c - gravar)',(1,450),
                   cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),1,cv.LINE_AA)
        
        
        imagem_canny = cv.cvtColor(imagem_canny,cv.COLOR_GRAY2BGR)
        
        imagem_cinza = cv.cvtColor(imagem_cinza,cv.COLOR_GRAY2BGR)
        
        fila_cima = cv.hconcat([imagem_canny,imagemMatrix]) 
        
        fila_baixo = cv.hconcat([frame,imagem_cinza])
        
        imagem_final = cv.vconcat([fila_cima,fila_baixo])
        
        
        cv.imshow('Resultado',imagem_final)
        if grava: 
            
            filme.write(imagem_final)

    key = cv.waitKey(50)
    
    if key == ord('q'):
        break
    elif key == ord('c'):
        inicializaFilme()
        grava = True
    elif key == ord('a'):
        cv.imwrite('./prezadix.jpg',imagemMatrix)
        cv.imwrite('./prezatoon.jpg',invertida)
        



cv.destroyAllWindows()
cap.release()
filme.release()
