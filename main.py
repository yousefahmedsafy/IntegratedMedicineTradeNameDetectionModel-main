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














app = FastAPI()







    
# Optional: Enable CORS if you need to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your allowed origins, or ["http://localhost", ...] for specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)






# @app.on_event("startup")
# async def startup_event():
#     print("Starting up the application...")
    
# @app.on_event("shutdown") 
# async def shutdown_event():
#     print("Shutting down the application...")



@app.post("/GetDetectedMedicineInfo/")
async def get_image_dimensions(allowStripRecognition:bool = False ,file: UploadFile = File(...)):
    image = None
    try:
        # Read image file as bytes
        image_bytes = await file.read()
        
        # Convert image bytes to numpy array
        image_array = np.frombuffer(image_bytes, np.uint8)
        
        # Decode numpy array to image using OpenCV
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR) 
        
    except:
        return {"Id":None, "MedicineName":None ,"TypeName": MedicineClasses.Rejected.name, "TypeValue": MedicineClasses.Rejected.value, "OcrExtractedName":None}


    try:
        classifyValue = classfiy(image,model,0.8)
    except:
        return {"Id":None, "MedicineName":None ,"TypeName": MedicineClasses.Rejected.name, "TypeValue": MedicineClasses.Rejected.value, "OcrExtractedName":None}


    
    
    try:
        searchedMedicine = [None,None]
        
        
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
    except:
        return {"Id":None, "MedicineName":None ,"TypeName": classifyValue.name, "TypeValue": classifyValue.value, "OcrExtractedName":None}




@app.get("/")
async def root():
    return {"message": "Welcome to the Image Dimensions API!"}