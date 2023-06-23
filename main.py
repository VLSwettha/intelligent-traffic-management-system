# TechVidvan Vehicle counting and Classification

# Import necessary packages

import cv2
import csv
import collections
import numpy as np
from tracker import *
import requests
import imutils
import warnings
import pyfirmata
import time

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('video.mp4')
input_size = 320

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line_position = 225   
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# Function for count vehicle
def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames 
    detected_classNames = []
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score 
        cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)

def from_static_image(image):
    img = cv2.imread(image)

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    # Set the input of the network
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
    outputs = net.forward(outputNames)

    # Find the objects from the network output
    try:
        postProcess(outputs,img)
        isVechicle = True
    except:
        totalCount = 0
        isVechicle = False
    # count the frequency of detected classes
    if isVechicle:
        frequency = collections.Counter(detected_classNames)       
        # Draw counting texts in the frame
        cv2.putText(img, "Car:        "+str(frequency['car']), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  "+str(frequency['motorbike']), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(frequency['bus']), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      "+str(frequency['truck']), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    
        carCout = str(frequency['car']) 
        motorbikeCout = str(frequency['motorbike'])
        busCount = str(frequency['bus'])
        truckCount = str(frequency['truck'])

        totalCountofVechicles = int(carCout)+int(motorbikeCout)+int(busCount)+int(truckCount)
        print("Total Count = ",totalCountofVechicles)
        
        def GST():
            def numerator(no_vehicles, avg_time):
                eqn1 = (no_vehicles * avg_time)
                return eqn1
            c,b,t,a=0,0,0,0
            if int(carCout) > 0:
                if int(carCout)<=1:              
                    no_vehicles = int(carCout)
                    avg_time = 30
                    c = numerator(no_vehicles, avg_time)
                else:
                    no_vehicles = int(carCout)
                    avg_time = 30 + (no_vehicles-1)*3
                    c = numerator(no_vehicles, avg_time)

            if int(motorbikeCout) > 0:
                if int(motorbikeCout)<=1:
                    no_vehicles = int(motorbikeCout)
                    avg_time = 21
                    b = numerator(no_vehicles, avg_time)
                else:
                    no_vehicles = int(motorbikeCout)
                    avg_time = 21 + (no_vehicles-1)*2
                    b = numerator(no_vehicles, avg_time)

            if int(truckCount) > 0:
                if int(truckCount)<=1:
                    no_vehicles = int(truckCount)
                    avg_time = 54
                    t = numerator(no_vehicles, avg_time)
                else:
                    no_vehicles = int(truckCount)
                    avg_time = 54 + (no_vehicles-1)*6
                    t = numerator(no_vehicles, avg_time)

            if int(busCount) > 0:
                if int(busCount)<=1:
                    no_vehicles = int(busCount)
                    avg_time = 60
                    a = numerator(no_vehicles, avg_time)
                else:
                    no_vehicles = int(busCount)
                    avg_time = 60 + (no_vehicles-1)*6
                    a = numerator(no_vehicles, avg_time)

            gst = (c + b + t + a) // 3

            return gst

        totalCount = GST()

        # print("Total car Count", carCout)
        # print("Total motorbike Count", motorbikeCout)
        # print("Total bus Count", busCount)
        # print("Total truck Count", truckCount)
        # print("Total vehicles", totalCount)

    # cv2.imshow("image", img)

    # cv2.waitKey(0)

    # save the data to a csv file
    # with open("static-data.csv", 'a') as f1:
    #     cwriter = csv.writer(f1)
    #     cwriter.writerow([image, frequency['car'], frequency['motorbike'], frequency['bus'], frequency['truck']])
    # f1.close()

    return totalCount

def getImageFromCam(camUrl, cam1Url):
    headers = {
    'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
    }
    img_resp = requests.get(camUrl, headers=headers, verify=False)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    # cv2.imshow("Android_cam", img)
    # cv2.waitKey()
    cv2.imwrite(cam1Url, img)
    image_file = cam1Url
    totalCount = from_static_image(image_file) 
    return totalCount

if __name__ == '__main__':  

    comport='COM12'

    board=pyfirmata.Arduino(comport)

    y1 = board.get_pin('d:13:o')
    g1 = board.get_pin('d:12:o')
    r1 = board.get_pin('d:11:o')

    r2 = board.get_pin('d:10:o')
    y2 = board.get_pin('d:9:o')
    g2 = board.get_pin('d:8:o')

    y3 = board.get_pin('d:4:o')
    g3 = board.get_pin('d:3:o')
    r3 = board.get_pin('d:2:o')


    r4 = board.get_pin('d:7:o')
    y4 = board.get_pin('d:6:o')
    g4 = board.get_pin('d:5:o')

    cam1 = 'https://192.168.199.44:8080/shot.jpg'
    cam1Url = './cam1/shotOnCam1.png'

    cam2 = 'https://192.168.199.41:8080/shot.jpg'
    cam2Url = './cam1/shotOnCam2.png'

    cam3 = 'https://192.168.199.50:8080/shot.jpg'
    cam3Url = './cam1/shotOnCam3.png'

    cam4 = 'https://192.168.199.89:8080/shot.jpg'
    cam4Url = './cam1/shotOnCam4.png' 
    
    # def yellow():
    #     while(gstTimeFromCam1==0 and gstTimeFromCam2==0 and gstTimeFromCam3==0 and gstTimeFromCam4==0):
    #         y1.write(1)
    #         y2.write(1)
    #         y3.write(1)
    #         y4.write(1)
    #         time.sleep(1)
    #         y1.write(0)
    #         y2.write(0)
    #         y3.write(0)
    #         y4.write(0)
    #         time.sleep(1)
            # gstTimeFromCam1 = getImageFromCam(cam1, cam1Url)
            # gstTimeFromCam2 = getImageFromCam(cam2, cam2Url) 
        # y1.write(1)
        # r2.write(1)
        # r3.write(1)
        # r4.write(1)
    # yellow()
    while (True):
        gstTimeFromCam1 = getImageFromCam(cam1, cam1Url)
        gstTimeFromCam2 = getImageFromCam(cam2, cam2Url)
        gstTimeFromCam3 = getImageFromCam(cam3, cam3Url)
        gstTimeFromCam4 = getImageFromCam(cam4, cam4Url)

        # print("Count from cam1 : ", gstTimeFromCam1)
        # print("Count from cam2 : ", gstTimeFromCam2)
        # print("Count from cam3 : ", gstTimeFromCam3)
        # print("Count from cam4 : ", gstTimeFromCam4)

        threshold=60
        lane={}
        cam_light={}
        GST_sec=0

        def first_green():

            global lane
            global cam_light
            global GST_sec
            lane={'Lx':getImageFromCam(cam1, cam1Url), 'Ly':getImageFromCam(cam2, cam2Url), 'Lz':getImageFromCam(cam3, cam3Url), 'Lw':getImageFromCam(cam4, cam4Url)}
            cam_light={'Lx':'RGY1', 'Ly': 'RGY2', 'Lz': 'RGY3', 'Lw': 'RGY4'}

            lane={k: v for k, v in sorted(lane.items(), key=lambda item: item[1],reverse=True)}
                
            print(lane)
            print(cam_light)

            print(next(iter(lane)))


            if(int(lane[list(lane)[0]]) > threshold):
                print("set the corresponding lane green for 60 sec")
                
                s_num=cam_light[next(iter(lane))]
                light(s_num, 1)
                
                lane.pop(list(lane)[0])
                print(lane)    
            else: 
                GST_sec = lane[list(lane)[0]]
                print(GST_sec)
                print("set the corresponding lane green for ",GST_sec,"seconds")  
                
                s_num=cam_light[next(iter(lane))]
                light(s_num,0)
                
                lane.pop(list(lane)[0]) 
                print(lane)
            
            
            
        def second_green():
            
            print("FOR 2nd round")  
            global GST_sec
            global lane
            global cam_light
            lane={k: v for k, v in sorted(lane.items(), key=lambda item: item[1],reverse=True)}
            print(lane)

            print(next(iter(lane)))

            global d
            d =int(lane[list(lane)[1]]) - int(lane[list(lane)[2]])
            print("d= ",d)

            if d>5:
                
                GST_sec = lane[list(lane)[-1]]
                
                if GST_sec>threshold:
                    print("set the corresponding lane green for 60 sec")
                    
                    s_num=cam_light[list(lane.keys())[-1]]
                    light(s_num, 1)
                    
                    lane.pop(list(lane)[-1]) 
                    print(lane)
                else:
                    print("set the corresponding lane green for ",GST_sec,"seconds")
                    
                    s_num=cam_light[list(lane.keys())[-1]]
                    light(s_num, 0)
                    
                    lane.pop(list(lane)[-1])
                    print(lane) 
            else:
                if(int(lane[list(lane)[0]]) > threshold):
                    print("set the corresponding lane green for 60 sec")
                    
                    s_num=cam_light[next(iter(lane))]
                    light(s_num, 1)
                    
                    lane.pop(list(lane)[0])
                    print(lane)    
                else: 
                    GST_sec = lane[list(lane)[0]]
                    print(GST_sec)
                    print("set the corresponding lane green for ",GST_sec,"seconds") 
                    s_num=cam_light[next(iter(lane))]
                    light(s_num, 0)
                    
                    lane.pop(list(lane)[0]) 
                    print(lane)

        def final_green():
            global GST_sec
            global lane
            global cam_light

            lane={k: v for k, v in sorted(lane.items(), key=lambda item: item[1],reverse=True)}

            print(lane)

            print(next(iter(lane)))

            if(int(lane[list(lane)[0]]) > threshold):
                print("set the corresponding lane green for 60 sec")
                
                s_num=cam_light[next(iter(lane))]
                light(s_num, 1)
                
                lane.pop(list(lane)[0])
                print(lane)    
            else: 
                GST_sec = lane[list(lane)[0]]
                print(GST_sec)
                print("set the corresponding lane green for ",GST_sec,"seconds") 
                
                s_num=cam_light[next(iter(lane))]
                light(s_num, 0)
                
                lane.pop(list(lane)[0]) 
                print(lane)
            

            print("For 4th round")

            if(int(lane[list(lane)[0]]) > threshold):
                print("set the corresponding lane green for 60 sec")
                
                s_num=cam_light[next(iter(lane))]
                light(s_num, 1)
                
                lane.pop(list(lane)[0])
                print(lane)    
            else: 
                GST_sec = lane[list(lane)[0]]
                print(GST_sec)
                print("set the corresponding lane green for ",GST_sec,"seconds") 
                
                s_num=cam_light[next(iter(lane))]
                light(s_num, 0)
                
                lane.pop(list(lane)[0]) 
                print(lane)

        def light(s_num,x):
            global GST_sec
            if s_num=='RGY1':
                if x==1:
                    print("light 1st lane for 60sec")
                    r1.write(0)
                    r2.write(1)
                    r3.write(1)
                    r4.write(1)
                    g1.write(1)
                    time.sleep(60)
                    g1.write(0)
                    r1.write(1)             
                else:
                    print("light first lane for GSTsec")
                    r1.write(0)
                    r2.write(1)
                    r3.write(1)
                    r4.write(1)
                    g1.write(1)
                    time.sleep(GST_sec)
                    g1.write(0)
                    r1.write(1) 
            if s_num=='RGY2':
                if x==1:
                    print("light 2nd lane for 60sec")
                    r1.write(1)
                    r2.write(0)
                    r3.write(1)
                    r4.write(1)
                    g2.write(1)
                    time.sleep(60)
                    g2.write(0)
                    r2.write(1)
                else:
                    print("light 2nd lane for GSTsec")
                    print("light first lane for GSTsec")
                    r1.write(1)
                    r2.write(0)
                    r3.write(1)
                    r4.write(1)
                    g2.write(1)
                    time.sleep(GST_sec)
                    g2.write(0)
                    r2.write(1) 
            if s_num=='RGY3':
                if x==1:
                    print("light 3rd lane for 60sec")
                    r1.write(1)
                    r2.write(1)
                    r3.write(0)
                    r4.write(1)
                    g3.write(1)
                    time.sleep(60)
                    g3.write(0)
                    r3.write(1) 
                else:
                    print("light 3rd lane for GSTsec")
                    print("light first lane for GSTsec")
                    r1.write(1)
                    r2.write(1)
                    r3.write(0)
                    r4.write(1)
                    g3.write(1)
                    time.sleep(GST_sec)
                    g3.write(0)
                    r3.write(1) 
            if s_num=='RGY4':
                if x==1:
                    print("light 4th lane for 60sec")
                    r1.write(1)
                    r2.write(1)
                    r3.write(1)
                    r4.write(0)
                    g4.write(1)
                    time.sleep(60)
                    g4.write(0)
                    r4.write(1) 
                else:
                    print("light 4th lane for GSTsec")
                    print("light first lane for GSTsec")
                    r1.write(1)
                    r2.write(1)
                    r3.write(1)
                    r4.write(0)
                    g4.write(1)
                    time.sleep(GST_sec)
                    g4.write(0)
                    r4.write(1) 
                
        # x=0 for normal GST_sec and x=1 for 60sec
                
        def traffic_signal():
            while (True):
                first_green()
                second_green()
                final_green()


        traffic_signal()