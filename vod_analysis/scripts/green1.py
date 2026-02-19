import cv2
import numpy as np
import json

with open('hsv_config.json', 'r') as f:
    config = json.load(f)
def nothing(x):
    pass

img = cv2.imread('3.png')
img = cv2.resize(img, (0,0), fx=3, fy=3) 

cv2.namedWindow('green filter params')
lower_green = np.array(config['lower_green'])
upper_green = np.array(config['upper_green'])
#tune filter
#Hue 35-85, Sat 50-255, Val 50-255
cv2.createTrackbar('Low H', 'green filter params', 20, 179, nothing)
cv2.createTrackbar('Low S', 'green filter params', 45, 255, nothing)
cv2.createTrackbar('Low V', 'green filter params', 38, 255, nothing)
cv2.createTrackbar('High H', 'green filter params', 85, 179, nothing)
cv2.createTrackbar('High S', 'green filter params', 255, 255, nothing)
cv2.createTrackbar('High V', 'green filter params', 255, 255, nothing)

while True:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos('Low H', 'green filter params')
    l_s = cv2.getTrackbarPos('Low S', 'green filter params')
    l_v = cv2.getTrackbarPos('Low V', 'green filter params')
    h_h = cv2.getTrackbarPos('High H', 'green filter params')
    h_s = cv2.getTrackbarPos('High S', 'green filter params')
    h_v = cv2.getTrackbarPos('High V', 'green filter params')
    
    lower_green = [l_h, l_s, l_v]
    upper_green =[h_h, h_s, h_v]

    mask = cv2.inRange(hsv,np.array(lower_green),np.array(upper_green))
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('green mask', mask)
    cv2.imshow('filtered frame', result)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        config_data = {"lower_green": lower_green, "upper_green": upper_green}
        with open("hsv_config.json",  'w') as f:
            json.dump(config_data,f,indent=4)
        print("values saved to hsv_config.json")
    
    elif key == ord('q'):
        break


cv2.destroyAllWindows()
