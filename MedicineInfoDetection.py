import numpy as np
import cv2 as cv
import re
from ImageProcessing import PreProcessing2,PreProcessing3,crop_image_by_4_points
from OpticalCharacterRecognition import CreatePaddlePipline, PaddleOpticalCharacterRecognition,PaddleOCR_TextDetection
import math






def RotateTitletoTop(image, DetailsOfLargestWord):

    """
    take arguments :
    1- image
    2- DetailsOfLargestWord ==> (startX, startY, endX, endY, Lenght, width)

    return :
    1-rotated_image)
    """


    (img_hieght, img_width) = image.shape[:2]

    #2nd way need to get you need coordinate of medicine pack in image....



    # startX, startY, endX, endY, length, width = DetailsOfLargestWord

    p1,_,p3,_ = DetailsOfLargestWord
    startX =p1[0]
    startY = p1[1]
    endX =p3[0]
    endY = p3[1]


    length = np.abs(endY-startY)
    width = np.abs(endX-startX)


    if width > length :
        #image is not need to rotate or need to rotate 180 deg
        X = startY
        Y = img_hieght - endY

        if(X>Y):
            print("DetectRotation180Degree")
            rotated_image = cv.rotate(image,cv.ROTATE_180)
            return  rotated_image

        return image
    else :
        #image is need to rotate 90 or 270 deg

        X = startX
        Y = img_width - endX

        if(X<=Y):
            print("DetectRotation90Degree")
            rotated_image = cv.rotate(image,cv.ROTATE_90_CLOCKWISE)
            return rotated_image
        else:
            print("DetectRotation270Degree")
            rotated_image = cv.rotate(image,cv.ROTATE_90_COUNTERCLOCKWISE)
            return rotated_image

def hasSubstring(list1,list2):
    # Example usage:
    # list1 = ["apple", "banana", "orange"]
    # list2 = ["app", "grape", "cherry"]
    for itemL1 in list1:
        for itemL2 in list2:
            if itemL2.lower() in itemL1.lower():
                return True
    return False

def CheckIfHasReadedKeyword(words):
    # words with with correct text orientation
    words = [word[0].lower() for word in words if abs(word[1][3][1] - word[1][0][1])<abs(word[1][2][0] - word[1][3][0])]
    
    Keywords = ['capsules','capsule','tablets','Pills','Caplets','Oral','Drops','Injectables','Cream','mg','gm','ml','tape','pharma']
    Regex = r'\b(\d+(\.\d+)?)\s*(mg|g|gm|capsules|tablets|caplets|ml)\b'

    for i in range(len(words)):    
        # if (re.findall(Regex , words[i], flags=re.IGNORECASE)) or any(element in Keywords for element in words) or hasSubstring(words,Keywords):
        if (re.findall(Regex , words[i], flags=re.IGNORECASE)):
            return True  
    
    if hasSubstring(words,Keywords):
        return True    
    return False


def PaddleOCR_TestSmart(rotated_image, scorePoints_textDetection, PaddlePipline):
    MedicineDetails = ["","","","",""]
    try:
       scorePoints1, *MedicineDetails = PaddleOpticalCharacterRecognition(rotated_image, PaddlePipline)
              
        
    except:
            MedicineDetails[5]=["",[[0,0],[0,0],[0,0],[0,0]]]
        
        
    if not CheckIfHasReadedKeyword(MedicineDetails[5]) or scorePoints1 < scorePoints_textDetection :
        rotated_image_copy = cv.rotate(rotated_image,cv.ROTATE_180) 
        try: 
            scorePoints2, *MedicineDetails2 = PaddleOpticalCharacterRecognition(rotated_image_copy, PaddlePipline)
           
        except:
            MedicineDetails[5]=["",[[0,0],[0,0],[0,0],[0,0]]]
            
        # if CheckIfHasReadedKeyword(MedicineDetails[3]) or scorePoints2 >= scorePoints_textDetection:
        #     MedicineName,medicine_conc,_,words,allWordsReturned, allWordsTitle_Coordinate,largest_word_Name = MedicineDetails
        #     return rotated_image,MedicineName,medicine_conc,allWordsReturned,largest_word_Name  

        #you dont need to check  any more you already read 2nd time ... check the largest score 1st or 2nd time??
        if CheckIfHasReadedKeyword(MedicineDetails2[5]) or scorePoints2 > scorePoints1:
            MedicineName,medicine_conc,_,words,allWordsReturned, allWordsTitle_Coordinate,largest_word_Name = MedicineDetails2
            return rotated_image_copy,MedicineName,medicine_conc,allWordsReturned,largest_word_Name  
    
    MedicineName,medicine_conc,_,words,allWordsReturned, allWordsTitle_Coordinate,largest_word_Name = MedicineDetails
    return rotated_image,MedicineName,medicine_conc,allWordsReturned,largest_word_Name  



def angle_between_points(p1, p2):
    """_summary_

    Args:
        p1 (x1,y1): _description_
        p2 (x2,y2): _description_

    Returns:
        angle of slope : to rotate imageby it.
    """



    # Calculate the differences in x and y coordinates

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Calculate the angle using atan2, and convert from radians to degrees
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # Ensure the angle is between 0 and 360 degrees
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg



def rotate_imageExpand(image, angle,rotationCenter = None):
    
    height, width = image.shape[:2]

    if rotationCenter == None:
        # Get image dimensions
        rotationCenter = (width / 2, height / 2)
        
    # Calculate the rotation matrix
    rotation_matrix = cv.getRotationMatrix2D(rotationCenter, angle, 1)

    # Determine the new size of the image
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))

    # Adjust the rotation matrix to translate the image to the center
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # Perform the rotation using the warpAffine function
    rotated_image = cv.warpAffine(image, rotation_matrix, (new_width, new_height))

    return rotated_image









def DetectMedicineTradeName(image,PaddlePipline):
    

    ordered_corners_clockwise,small_img,nobg_img,imggray,thresholded,small_img_copy,drawed_Img_Corners = PreProcessing2(image)



    #angle of all medicine pack contours.......
    angle = 0
    if((ordered_corners_clockwise[2][0]-ordered_corners_clockwise[3][0])>(ordered_corners_clockwise[1][0]-ordered_corners_clockwise[0][0])):   
        angle = angle_between_points(ordered_corners_clockwise[3],ordered_corners_clockwise[2])
    else:
        angle = angle_between_points(ordered_corners_clockwise[0],ordered_corners_clockwise[1])
        
    angleImage1 = rotate_imageExpand(nobg_img,angle)
    originalImageAfterRotation1 = rotate_imageExpand(small_img,angle)

    #crop image ..
    #for cropping image only
    ordered_corners_clockwise,_,_,_ = PreProcessing3(angleImage1)
    croppedImage = crop_image_by_4_points(angleImage1,ordered_corners_clockwise)
    croppedImageOriginal1 = crop_image_by_4_points(originalImageAfterRotation1,ordered_corners_clockwise)


    #text detection to get  score points ,, coordinates of largest text to rotate image according it (largest is top)
    scorePoints_textDetection,textDetected_Img, PointsArray_of_largest_word,NameOfLargestWord = PaddleOCR_TextDetection(croppedImage,PaddlePipline)


    #rotate image using rotation center is >> center Largest text
    angle = angle_between_points(PointsArray_of_largest_word[3],PointsArray_of_largest_word[2])

    #rotate image using rotation center is >> center Largest text  >> use center of image instead
    angleImage2 = rotate_imageExpand(croppedImage,angle)
    originalImageAfterRotation2 = rotate_imageExpand(croppedImageOriginal1,angle)


    ordered_corners_clockwise,_,_,_ = PreProcessing3(angleImage2)
    croppedImage = crop_image_by_4_points(angleImage2,ordered_corners_clockwise)
    croppedImageOriginal2 = crop_image_by_4_points(originalImageAfterRotation2,ordered_corners_clockwise)




    rotated_image = RotateTitletoTop(croppedImageOriginal2,PointsArray_of_largest_word)


    #------- Paddle OCR Test ---------
    medicineDetails = PaddleOCR_TestSmart(rotated_image ,scorePoints_textDetection, PaddlePipline)
    #------- Paddle OCR Test ---------

    # largestwordName is not found : takecare..

    Final_image,MedicineName,medicine_conc,allWordsReturned,largest_word_Name = medicineDetails
    return MedicineName




