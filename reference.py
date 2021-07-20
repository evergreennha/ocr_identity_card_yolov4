from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import os
import json

# update dict_information after ocr text
def udate_dict(list_infor):
    dict_infor = {}
    for i in range(len(list_infor)):
        if(list_infor[i][0] == list_infor[i-1][0]):
            list_infor[i][1] = list_infor[i-1][1]+", "+list_infor[i][1]
    for i in range(len(list_infor)):
        dict_infor.update({list_infor[i][0]:list_infor[i][1]})
    return dict_infor

# yolov4 detect text region by using cv2
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
net = cv2.dnn.readNet("yolov4-custom_last.weights", "yolov4-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
img = cv2.imread('test1.jpg')
classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
list_infor = []
path = "E:/machine_learning/id_yolov4/data_save/"


# ocr model
config = Cfg.load_config_from_name('vgg_transformer')
config['export'] = 'transformerocr.pth'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False


index = 0
for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = "%s : %f" % (class_names[classid[0]], score)
    cv2.rectangle(img, box, color, 2)     # box = (x,y,h,w)
    cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    imgcrop = img[box[1]:(box[1]+box[3]),box[0]:(box[0]+box[2])]
    path_data_reference = os.path.join(path,class_names[classid[0]])
    os.chdir(path_data_reference)
    cv2.imwrite("box"+str(index)+str(class_names[classid[0]])+".jpg",imgcrop)

    detector = Predictor(config)
    image_orc = Image.open("box"+str(index)+str(class_names[classid[0]])+".jpg")

    tp = [class_names[classid[0]],detector.predict(image_orc)]
    list_infor.append(tp)
    
    index = index+1
dict_infor = udate_dict(list_infor)
json_type = json.dumps(dict_infor,ensure_ascii=False,indent=4)
print(json_type)

# cv2.imwrite("3.jpg", img)