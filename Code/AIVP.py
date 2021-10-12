import cv2
import math
import numpy as np
import time
import pyautogui

class virtual_mouse():
    def __init__(self):
        # Necessary attributes for the cursor movement
        pyautogui.FAILSAFE = False
        self.SCREEN_X, self.SCREEN_Y = pyautogui.size()
        self.click = self.click_msg = self.scroll_start = None

    # Helper function to retrieve the contour that has the biggest area
    # Ideally, this should be the hand, thus the background should be relatively empty
    # Takes the contours as inputs, selects the biggest one
    def getMaxContours(self, contours):
        maxIndex = 0
        maxArea = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)

            if area > maxArea:
                maxArea = area
                maxIndex = i
        return contours[maxIndex]

    # Helper function for the retrieval and drawing of the contours
    def draw_contours(self, cnt, img, thresh):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 0)
        hull = cv2.convexHull(cnt)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        # count_defects = 0
        cv2.drawContours(thresh, self.contours, -1, (0, 255, 0), 3)
        return defects, drawing

    # Helper function to identify convexity defects, that enables to identify how many fingers are shown
    def convexity_defects(self, img, cnt, defects):
        count_defects = 0
        used_defect = None
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
            if angle <= 90:
                count_defects += 1
            cv2.circle(img, far, 5, [0, 0, 255], -1)
            cv2.line(img, start, end, [0, 255, 0], 2)
            if count_defects == 1 and angle <= 90:
                used_defect = {"x": start[0], "y": start[1]}

        return used_defect, count_defects

    def mouse(self, crop_img, used_defect, count_defects):
        if used_defect is not None:
            best = used_defect
            if count_defects == 1:
                x = best['x']
                y = best['y']

                display_x = x
                display_y = y

                if self.scroll_start is not None:
                    M_START = (x, y)
                    x = x - self.scroll_start[0]
                    y = y - self.scroll_start[1]
                    x = x * (self.SCREEN_X / self.CAMERA_X)
                    y = y * (self.SCREEN_Y / self.CAMERA_Y)
                    self.scroll_start = M_START
                    # print("X: " + str(x) + " Y: " + str(y))
                    pyautogui.moveRel(x, y)
                else:
                    self.scroll_start = (x, y)

                cv2.circle(crop_img, (display_x, display_y), 5, [255, 255, 255], 20)
            elif count_defects == 4 and self.click is None:
                self.click = time.time()
                pyautogui.click()
                self.click_msg = "LEFT CLICK"
            elif count_defects == 3 and self.click is None:
                self.click = time.time()
                pyautogui.rightClick()
                self.click_msg = "RIGHT CLICK"


            elif count_defects == 2 and self.click is None:
                self.click = time.time()
                pyautogui.press("down")
                self.click_msg = "SCROLL DOWN"
        else:
            self.scroll_start = None

        if self.click is not None:
            cv2.putText(self.img, self.click_msg, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
            if self.click < time.time():
                self.click = None

        cv2.putText(self.img, "Fingers: " + str(count_defects+1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        cv2.imshow('Gesture', self.img)


    def main(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():

            # Take input from web-camera
            self.ret, self.img = cap.read()
            self.CAMERA_X, self.CAMERA_Y, channels = self.img.shape

            # Invert the image for mirror-like use for mouse gestures
            self.img = cv2.flip(self.img, 1)

            # Backup the original image before processing
            self.crop_img = self.img

            # Conversion to Gray scale
            grey = cv2.cvtColor(self.crop_img, cv2.COLOR_BGR2GRAY)

            # Applying Gaussian blur, removes noise from original input image
            value = (35, 35)
            blurred = cv2.GaussianBlur(grey, value, 0)
            cv2.imshow('Gaussian Blur', blurred)

            # Applying threshold, using Otsu's Binarization Method
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cv2.imshow("Otsu's Thresholding", thresh)

            # Retrieving the contour which correspond to the edges of the hand shown
            self.contours, self.hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # Retrieving the contour with the biggest area, that should be the hand ideally
            max_cnt = self.getMaxContours(self.contours)

            # Drawing of the contours and depicting the image
            defects, drawing = self.draw_contours(max_cnt, self.crop_img, thresh)
            cv2.imshow('Drawing', drawing)

            # Convexity defects
            used_defect, count_defects = self.convexity_defects(self.img, max_cnt, defects)

            # Mouse interactions
            self.mouse(self.crop_img, used_defect, count_defects)

            # Assigning keyboard keys to different roles
            k = cv2.waitKey(1) & 0xFF

            # Pressing "q" exits the application, is rather important because of the camera usage
            if k == ord("q"):
                break

            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    v_m = virtual_mouse()
    v_m.main()