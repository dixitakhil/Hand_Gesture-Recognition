import cv2
from keras.models import load_model
import numpy as np
import tkinter as tk   #gui module
import threading   #To have multithread for tkinter and opencv

model = load_model('model.h5')
last_alpabet = ''  #Last alphabet predicted by model
text = ''   #Existing text

root = tk.Tk()   #Tkinter code
root.title('Predicted handsigns decipher')
t = tk.Text(root, height=2, width=30)
t.pack()



def main():
    cap = cv2.VideoCapture(0)
    x, y, w, h = 300, 50, 350, 350

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))
        res = cv2.bitwise_and(img, img, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=2)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)

        thresh = thresh[y:y + h, x:x + w]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w1, h1 = cv2.boundingRect(contour)
                nImage = thresh[y:y + h1, x:x + w1]
                nImage = cv2.resize(nImage, (50, 50))
                probab, p_class = model_predict(model, nImage)
                print(p_class, probab)
                decipher_and_print(p_class, img)  #Decipher the class to the correct output and print

        x, y, w, h = 300, 50, 350, 350
        cv2.imshow("Output", img)
        cv2.imshow("Threshold and Contour", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break


def decipher_and_print(value, img):
    Alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'X', 'Y', 'Z']
    chosen_alphabet = Alphabets[value - 1]
    global text
    global last_alpabet
    if not last_alpabet == chosen_alphabet:
        text = text + chosen_alphabet
        last_alpabet = chosen_alphabet

    print(text)
    cv2.putText(img, text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
    global t

    t.insert(tk.END,text)


def model_predict(model, image):
    processed = process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


model_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
thread=threading.Thread(target=main,args=())
thread.start()
t.mainloop()