import torch
import pathlib
import cv2
from enum import Enum 


class MedicineClasses(Enum):
    Rejected = 0,
    Strip = 1,
    package = 2

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# Load the YOLOv5 model with custom weights
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt',force_reload=True,trust_repo=True)








def classfiy(image,model,conf=0.8):
    img= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img= cv2.resize(img,(640,640))
    results = model(img)
    try:
        if results.pandas().xyxy[0] ['confidence'][0] and results.pandas().xyxy[0] ['confidence'][0]>conf:
            name = results.pandas().xyxy[0]['name'][0]
            if name == MedicineClasses.package.name: return MedicineClasses.package
            elif name == MedicineClasses.Strip.name: return MedicineClasses.Strip
            else : return MedicineClasses.Rejected
             
        else:
            return MedicineClasses.Rejected
    except Exception as e:
        # print(e)
        return MedicineClasses.Rejected
    
    
    
    
    
    
if __name__ == "__main__":

    print(classfiy(r"C:\Users\Islam\OneDrive\Desktop\download.jpeg"))
    
    


