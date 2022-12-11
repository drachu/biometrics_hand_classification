import cv2
import numpy as np
import math


def verify_veins(frame, frame_orig):
    roi=cv2.selectROI(frame)
    roi_cropped=frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    # cv2.destroyAllWindows()
    # skala szarości
    gray = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2GRAY)
    # wyrównanie histogramu
    equ_hist = cv2.equalizeHist(gray)
    # zaawansowany threshold - Ograniczona kontrastowa adaptacyjna korekcja histogramu
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    # zaaplikowanie wyrównanego histogramu
    cl1 = clahe.apply(equ_hist)
    # zwykły Gaussowski blur
    blur = cv2.GaussianBlur(cl1, (7, 7), 0)
    img_gray = blur
    cv2.imwrite('bartek.jpg', img_gray)
    cv2.imshow('frame', img_gray)
    cv2.waitKey(0)

    template = cv2.imread('template.jpg', 0)
    w_template, h_template = template.shape[::-1]
    img_gray = cv2.resize(img_gray, (w_template, h_template), cv2.INTER_AREA)

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)[0][0]
    cv2.putText(frame_orig, 'PODOBIENSTWO KUBY: ' + str(round(res, 2)), (0, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame_orig)
    cv2.waitKey(0)

    
frame = cv2.imread("images/bartek_otw_wew.jpg")
frame_orig = frame.copy()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mask = cv2.inRange(hsv, 110, 255)

#wyciągnięcie dobrego z obrazu
kernel = np.ones((3, 3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=3)
mask = cv2.bilateralFilter(mask,30,15,15)

#kontury
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#wybranie największego konturu
contour_max = max(contours, key = lambda x: cv2.contourArea(x))
#obliczenie obwodu
epsilon = 0.005*cv2.arcLength(contour_max, True)
#obwiednia konturu
hull = cv2.convexHull(contour_max)


#powierzchnia obwiedni
areahull = cv2.contourArea(hull)
#powierzchnia konturu
areacnt = cv2.contourArea(contour_max)
#obliczenie procentowej powierzchni poza konturem - powierzchnia nie obejmująca dłoni
arearatio = ((areahull-areacnt)/areacnt)*100

#aproksymacja punktów skrajnych konturu
approx = cv2.approxPolyDP(contour_max, epsilon, True)
#obwiednia punktów skrajnych konturu
hull = cv2.convexHull(approx, returnPoints=False)
#odległości punktów skrajnych wewnętrznych od obwiedni
defects = cv2.convexityDefects(approx, hull)
#zdefiniowane wystarczających odległości z kątami
possible_fingers=0

#iterowanie po odległościach punktów skrajnych wewnątrz dłoni od obwiedni
for i in range(defects.shape[0]):
    start, end, far, d = defects[i, 0]
    start_point = tuple(approx[start][0])
    far_point = tuple(approx[far][0])
    end_point = tuple(approx[end][0])

    #odległości znalezionego 'trójkąta'
    distance_between_fingers = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
    distance_first_finger = math.sqrt((far_point[0] - start_point[0]) ** 2 + (far_point[1] - start_point[1]) ** 2)
    distance_second_finger = math.sqrt((end_point[0] - far_point[0]) ** 2 + (end_point[1] - far_point[1]) ** 2)
    #obliczenie pola trójkąta
    s = (distance_between_fingers + distance_first_finger + distance_second_finger) / 2
    ar = math.sqrt(s * (s - distance_between_fingers) * (s - distance_first_finger) * (s - distance_second_finger))
    #obliczenie kąta trójkąta od wewnętrznego punktu
    angle = math.acos((distance_first_finger ** 2 + distance_second_finger ** 2 - distance_between_fingers ** 2) / (2 * distance_first_finger * distance_second_finger)) * 57

    #zdefiniowanie czy odcinek się nadaje
    if angle <= 90 and d>30:
        possible_fingers += 1
        #rysowanie wewnętrznego punktu
        cv2.circle(frame, far_point, 3, [255, 0, 0], -1)

    #rysowanie linii między możliwymi palcami
    cv2.line(frame, start_point, end_point, [0, 255, 0], 2)

possible_fingers+=1

font = cv2.FONT_HERSHEY_SIMPLEX
if possible_fingers==1:
    if arearatio<50:
        cv2.putText(frame, 'ZAMKNIETA', (0, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        cv2.waitKey(0)
    else:
        cv2.putText(frame, 'OTWARTA - ZAZANACZ ROI', (0, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        verify_veins(frame=frame, frame_orig=frame_orig)

elif possible_fingers>1:
    cv2.putText(frame, 'OTWARTA - ZAZNACZ ROI', (0, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    verify_veins(frame=frame, frame_orig=frame_orig)

else:
    cv2.putText(frame, 'ZAMKNIETA', (0, 50), font, 2, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    cv2.waitKey(0)




















