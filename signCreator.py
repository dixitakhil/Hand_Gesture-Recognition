import numpy as np   #Numpy is used to handle dimensional arrays
import cv2   #OpenCv
import os    #Os module to handle directories

im_x, im_y = 50, 50    #Specifying the size



def folderCreate(folder_name):    #Creates directory and folders
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def main(index):
    total_pics = 1400   #Taking 1400 pictures for each sign
    cap = cv2.VideoCapture(0)  #Start video capture
    x, y, w, h = 300, 50, 350, 350   #Rectangular window dimensions

    folderCreate("signs/" + str(index))  #Creating folder
    image_index = 0  #Index of the stored or taken images
    flag_capuring = False   #If contour size is less then it remains false
    frames = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))  #Acquiring required color tone
        res = cv2.bitwise_and(frame, frame, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=2)
        opening=cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel_square)

        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)
        thresh = thresh[y:y + h, x:x + w]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                image_index += 1
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                save_img = cv2.resize(save_img, (im_x, im_y))
                cv2.putText(frame, "Acquiring data please hold.", (30, 60), cv2.FONT_ITALIC, 2, (127, 255, 255))
                cv2.imwrite("signs/" + str(index) + "/" + str(image_index) + ".jpg", save_img)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(image_index), (60, 400), cv2.FONT_ITALIC, 4, (100, 160, 255))
        cv2.imshow("Acquire Data", frame)
        cv2.imshow("Main", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            if flag_capuring == False:
                flag_capuring = True
            else:
                flag_capuring = False
                frames = 0
        if flag_capuring == True:
            frames += 1
        if image_index == total_pics:
            break


index = input("Please enter the hand sign number::: ")
main(index)