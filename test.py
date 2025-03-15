
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # Optional for cross-origin requests
import cv2
import numpy as np
from paddleocr import PaddleOCR,draw_ocr
import asyncio
import time


import torch
import pathlib
import os


#***********************************************************
# Windows
# Open a Command Prompt or PowerShell and run:
# set KMP_DUPLICATE_LIB_OK=TRUE

# Set the environment variable to avoid the OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#***********************************************************


from LevenshteinAlgorithm import SearchMedicine

from MedicineInfoDetection import DetectMedicineTradeName

from YOLO_Classifier import classfiy, MedicineClasses      


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# Load the YOLOv5 model with custom weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt',force_reload=True,trust_repo=True)
# time.sleep(20)

PaddlePipline = PaddleOCR(rec_algorithm='CRNN',use_angle_cls=True)




def main(image,allowStripRecognition):
    searchedMedicine = [None,None]
    classifyValue = classfiy(image,model,0.8)
    
    
    if classifyValue == MedicineClasses.package :
        
        extractedName = DetectMedicineTradeName(image,PaddlePipline)       
        if extractedName is not None and extractedName != "":             
            searchedMedicine = SearchMedicine(extractedName)     
        return {"Id":searchedMedicine[0], "MedicineName":searchedMedicine[1] ,"TypeName": MedicineClasses.package.name, "TypeValue": MedicineClasses.package.value, "OcrExtractedName":extractedName}
    
        
    elif classifyValue == MedicineClasses.Strip:
        
        if allowStripRecognition:
            extractedName = DetectMedicineTradeName(image,PaddlePipline)       
            if extractedName is not None and extractedName != "":             
                searchedMedicine = SearchMedicine(extractedName)       
            return {"Id":searchedMedicine[0], "MedicineName":searchedMedicine[1] ,"TypeName": MedicineClasses.Strip.name, "TypeValue": MedicineClasses.Strip.value, "OcrExtractedName":extractedName}
        else:
            return {"Id":searchedMedicine[0], "MedicineName":searchedMedicine[1] ,"TypeName": MedicineClasses.Strip.name, "TypeValue": MedicineClasses.Strip.value, "OcrExtractedName":None}

    
    return {"Id":searchedMedicine[0], "MedicineName":searchedMedicine[1] ,"TypeName": MedicineClasses.Rejected.name, "TypeValue": MedicineClasses.Rejected.value, "OcrExtractedName":extractedName}

if __name__ == "__main__":


    path=r"C:\Users\Islam\OneDrive\Desktop\WhatsApp Image 2024-06-20 at 14.45.14_219797c0.jpg"

    image = cv2.imread(path)

    x = main(image,False)
    print(x)
