"""get_img controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera
import cv2
import numpy as np
from test import predict
#from test_convolve import predict
#from is_new import new_norm
from PIL import Image
import io

robot = Robot()
timestep = int(robot.getBasicTimeStep())
camera = Camera('rear camera')
#camera = robot.getDevice("left head camera")
Camera.enable(camera, timestep)

    
cv2.startWindowThread()
cv2.namedWindow("preview")

# Main loop:
# - perform simulation steps until Webots is stopping the controller
i = 0
prob_arr = []
class_arr = []
prob_avg = 0.9
thresh = prob_avg
last_class_num = 5
del_class = 0
prob_jump = 0
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    img = camera.getImage()
    #image = np.array(img)
    image = np.frombuffer(img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    image = image[200:500, 400:680, :]
    #print(image[10,28,:])
    
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, bin_img = cv2.threshold(im_gray, 150, 192, cv2.THRESH_BINARY)
    n_white_pix = np.sum(bin_img > 0)
    #print(n_white_pix, i)
    flag = 0
    flag_2 = 0
    if n_white_pix > 250:
        i = i+1
        flag = 1
        class_num, prob = predict(image)
        #print("Toy number: ", class_num , " probability :", prob)
        if i%20 != 0:
            prob_arr.append(prob)
            class_arr.append(class_num)
            if class_num != last_class_num:
                last_class_num = class_num
                del_class = del_class + 1
            if prob < 0.8:
                prob_jump = prob_jump + 1
        else:
            flag = 2
            prob_arr2 = np.array(prob_arr)
            prob_arr = []
            class_arr = []
    else:
        del_class = 0
        prob_jump = 0
        prob_arr = []
        class_arr = []
        
    if flag == 2:
        prob_avg = np.average(prob_arr2)
        if del_class > 6 and prob_jump > 10:
            flag_2 = 1
            #print("               ",del_class, prob_jump)
            del_class = 0
            prob_jump = 0
    if prob_avg > thresh and flag_2 == 0 and flag == 2:
        flag = 3
        print("Toy", class_num , " probability :", prob_avg) 
        #pass
    elif flag == 2 and prob_avg < thresh:
        print("NEW NEW NEW NEW NEW NEW")
        gray = np.float32(im_gray)
        dst = cv2.cornerHarris(gray,2,5,0.07)
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        
        image2 = np.copy(image)
        #image2.setflags(write=1)
        print(dst.shape)
        image2[dst>0.01*dst.max(),]=[0,0,255,255]
        num_corners = np.sum(dst > 0.01 * dst.max())
        print(num_corners)
        if num_corners < 1500:
            print("NEW Safe For children")
        else:
            print("NEW Dangerous for children")
        
        
   
    if flag_2 == 1:
        cv2.imshow("preview", image2)
        cv2.waitKey(timestep)
    else:
        cv2.imshow("preview", image)
        cv2.waitKey(timestep)
    
    pass
    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)

# Enter here exit cleanup code.
