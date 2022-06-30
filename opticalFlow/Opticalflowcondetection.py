import numpy as np
import cv2
import time
from Contruir_el_OptFlow import *

def main():
    x_prev = 0
    y_prev = 0
    cam = cv2.VideoCapture(0) #camara
    p = int(cam.get(3))
    l = int(cam.get(4))

    ret, prev = cam.read()
   
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = False
    cur_glitch = prev.copy()

    # Definimosun video que guardara el opticalflo que estemos estremeando
    out = cv2.VideoWriter('ResultadodeoFLOW.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30 , (p,l))
    
    while True:
        ret, img = cam.read()
        #vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 50, 20, 5, 5, 10.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        prevgray = gray
        cv2.imshow('flow', draw_flow(gray,flow))

        if show_hsv:
            gray1 = cv2.cvtColor(draw_hsv(flow), cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray1, 3 , 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours,hierachy= cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     

            



            # Encontramos el contorno mas grande y analizamos sus coordenadas
            if len(contours) > 0:
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt=contours[max_index]   

                # draw a bounding box/rectangle around the largest contour
                x,y,w,h = cv2.boundingRect(cnt)
                # si el x previo y y previo es menor quiere decir que está bajando sino está subiendo 
                
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                area = cv2.contourArea(cnt)
                if x_prev < x and y_prev < y:
                    cv2.putText(img, "Enter HAND", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    x_prev=x
                    y_prev=y
                elif x_prev > x and y_prev > y:
                    cv2.putText(img, "Exit HAND", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    x_prev=x
                    y_prev=y
                else:
                    None
                # print area to the terminal
                print(area)
            
                # add text to the frame
                # cv2.putText(img, "Largest Contour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

           
            cv2.imshow('th', thresh)
            cv2.imshow('odf',draw_hsv(flow))
            cv2.imshow('Image', img)
        
            out.write(draw_flow(gray,flow))
            
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # execute main
    main()









