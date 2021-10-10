from absl import flags
import sys
FLAGS=flags.FLAGS
FLAGS(sys.argv)

import  time
import numpy as np
import  cv2
import  matplotlib.pyplot as plt

import  tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import  transform_images
from yolov3_tf2.utils import  convert_boxes

from deep_sort import preprocessing
from  deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

classNames=[c.strip() for c in open('coco.names').readlines()]
yolo=YoloV3(classes=len(classNames))
yolo.load_weights('./weights/yolov3.tf')

maxCosineDistance=0.5
nnBudget=None
NMSMaxOverlap=0.8

modelFilename='model_data/mars-small128.pb'
encoder=gdet.create_box_encoder(modelFilename,batch_size=1)
metric=nn_matching.NearestNeighborDistanceMetric('cosine',maxCosineDistance,nnBudget)
tracker=Tracker(metric)

vid=cv2.VideoCapture('./data/video/test.mp4')

codec=cv2.VideoWriter_fourcc(*'XVID')
vFPS=int(vid.get(cv2.CAP_PROP_FPS))
vW,vH=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

output=cv2.VideoWriter('./data/video/results.avi',codec,vFPS,(vW,vH))

while True:
    _,img=vid.read()
    if img is None:
        print('Completed')
        break

    imgInvert=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgInvert=tf.expand_dims(imgInvert,0)
    imgInvert=transform_images(imgInvert,416)

    t1=time.time()

    boxes,scores,classes,nums=yolo.predict(imgInvert)

    classes=classes[0]
    names=[]
    for i in range(len(classes)):
        names.append(classNames[int(classes[i])])
    names=np.array(names)
    convertedBoxes=convert_boxes(img,boxes[0])
    features=encoder(img,convertedBoxes)

    detections=[Detection(bbox,score,class_name,feature) for bbox,score,class_name,feature in zip(convertedBoxes,scores[0],names,features)]

    boxes=np.array([d.tlwh for d in detections])
    scores=np.array([d.confidence for d in detections])
    classes=np.array([d.class_name for d in detections])

    indices=preprocessing.non_max_suppression(boxes,classes,NMSMaxOverlap,scores)
    detections=[detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap=plt.get_cmap('tab20b')
    colors=[cmap(i)[:3] for i in np.linspace(0,1,50)]

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update>1:
            continue
        bbox=track.to_tlbr()
        className=track.get_class()
        color=colors[int(track.track_id)%len(colors)]
        color=[i*255 for i in color]
        cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),color,3)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+len(className)+len(str(track.track_id))*10, int(bbox[1])), color, -1)
        cv2.putText(img,className+"-"+str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0,0.75,(255,255,255),2)

    fps=1./(time.time()-t1)
    cv2.putText(img,"FPS:{:.2f}".format(fps),(0,30),0,1,(255,0,0),2)
    cv2.resizeWindow('output',1024,768)
    cv2.imshow('output',img)
    output.write(img)

    if(cv2.waitKey(1)==ord('q')):
        break
vid.release()
output.release()
cv2.destroyAllWindows()
