import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_into_canny(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grey_img, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(img):
    height = img.shape[0]
    triangle = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, triangle, 255)
    masked_images = cv2.bitwise_and(img, mask)
    return masked_images

def display_lane_line(images , lines):
    line_image = np.zeros_like(images)
    if lines is not None:
        for line in lines:
            # no need to reshape 
            x1 , y1 , x2 , y2 = line.reshape(4)
            cv2.line(line_image , (x1 , y1) , (x2 , y2) , (255 , 0 , 0) , 10)
    return line_image


def make_coordinates(image , lines):
    slope , intercept = lines
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept) / slope) # <--- derived from linear equation
    x2 = int((y2 - intercept) / slope)
    return np.array([x1 , y1 , x2 , y2])

# this func will only highlight lane with one line
def average_line_detect(real_image , lane_line):
    left_fit = []
    right_fit = []

    for line in lane_line:
        x1 , y1 , x2 , y2 = line.reshape(4)
        slope , intercept = np.polyfit((x1 , x2) , (y1 , y2) , 1)

        if slope > 0:
            right_fit.append((slope , intercept))
        else:
            left_fit.append((slope , intercept))
    
    left_fit_average = np.average(left_fit , axis=0)
    right_fit_average = np.average(right_fit , axis=0)

    left_line = make_coordinates(real_image , left_fit_average)
    right_line = make_coordinates(real_image , right_fit_average)
    return np.array([left_line , right_line])

# image = cv2.imread("./road_img.jpeg")
# new_img = np.copy(image)
# canny = convert_into_canny(new_img)
# cropped_img = region_of_interest(canny)
# lane_line = cv2.HoughLinesP(cropped_img , 2 , np.pi/180 , 100 , np.array([]) , minLineLength=40 , maxLineGap=5)
#                             # IMG  ,  pixelsize , degree , threshold , placeholder
# average_line = average_line_detect(new_img , lane_line)
# lane_img = display_lane_line(new_img , average_line)
# combo_img = cv2.addWeighted(new_img , 0.8 , lane_img , 1 , 1)
# cv2.imshow("result", combo_img)
# cv2.waitKey(0)

cap = cv2.VideoCapture("./test2.mp4")
while(cap.isOpened()):
    _ , frame = cap.read()
    canny = convert_into_canny(frame)
    cropped_img = region_of_interest(canny)
    lane_line = cv2.HoughLinesP(cropped_img , 2 , np.pi/180 , 100 , np.array([]) , minLineLength=40 , maxLineGap=5)
                            # IMG  ,  pixelsize , degree , threshold , placeholder
    average_line = average_line_detect(frame , lane_line)
    lane_img = display_lane_line(frame , average_line)
    combo_img = cv2.addWeighted(frame , 0.8 , lane_img , 1 , 1)
    cv2.imshow("result", combo_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
