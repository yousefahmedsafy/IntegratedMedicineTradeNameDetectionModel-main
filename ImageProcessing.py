import numpy as np
import cv2 as cv
from rembg import remove
import matplotlib.pyplot as plt
import re
import math
import requests

# from MedicineOCR import OpticalCharacterRecognition

#****************functions****************
def GetCornerPoints(largestContour):
    perimeter = cv.arcLength(largestContour,closed=True)
    # get the perimeter of polygon by contour
    # are the polygon closed? T/F


    corners = cv.approxPolyDP(largestContour, 0.02*perimeter ,closed= True)
    #get min points that act as corners of polygon then calc the perimeter by this points 
    # and combare it with perimeter calculated by original contours...

    #acceptable error (epsilon)

    return np.squeeze(corners)
    #return list without unused []
    #before: 
    # [
    #     [[1,2]],
    #     [[1,2]],
    #     [[1,2]],
    # ]

    #after:
    # [
    #     [1,2],
    #     [1,2],
    #     [1,2],
    # ]

def order_points(points):
    # Calculate centroid
    centroid = [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]

    # Calculate angles of points relative to centroid
    angles = [math.atan2(p[1] - centroid[1], p[0] - centroid[0]) for p in points]

    # Sort points based on angles
    sorted_points = [p for _, p in sorted(zip(angles, points),reverse= True)]

    return sorted_points

def OrderCornerPointClockwise(points):
    rectangel = sorted(points, key=lambda point: point[0])
    
    left = rectangel[:2] 
    right = rectangel[-2:]
    
    
    top_left = min(left, key=lambda point: point[1])
    buttom_left = max(left, key=lambda point: point[1])
    
    
    
    top_right = min(right, key=lambda point: point[1])
    buttom_right = max(right, key=lambda point: point[1])
    
    
    rectangel = [top_left,top_right,buttom_right,buttom_left]
    return rectangel


def DrawPointsOnImage(img,points,imgName):
    if len(points) == 4 :
        p1,p2,p3,p4 = points   
        cv.circle(img,tuple(p1),50,(255,0,0),-1)
        cv.circle(img,tuple(p2),50,(255,0,255),-1)
        cv.circle(img,tuple(p3),50,(0,255,0),-1)
        cv.circle(img,tuple(p4),50,(0,255,255),-1)

    return img

def ApplyTopView(img,points):
    (tl,tr,br,bl) = points

    widthA = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
    widthB = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
    maxWidth = min(int(widthA),int(widthB))

    heightA = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
    heightB = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
    maxheight = min(int(heightA),int(heightB))


    dst = np.array([
        [0,0],
        [maxWidth +1 , 0],
        [maxWidth+1,maxheight+1],
        [0,maxheight+1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(points,dst)
    warrped = cv.warpPerspective(img,M,(maxWidth,maxheight))
    return warrped




def GetDeltaY(ArrayofPoints):
  dy=np.abs(ArrayofPoints[0][1]-ArrayofPoints[3][1])
  return dy

def GetDeltaX(ArrayofPoints):
  dx=np.abs(ArrayofPoints[0][0]-ArrayofPoints[1][0])
  return dx








def crop_image_by_4_points(image, points):
    # Convert points to numpy array
    pts = np.array(points, dtype=np.float32)
    
    # Find the bounding box
    x, y, w, h = cv.boundingRect(pts)
    
    # Crop the image using the bounding box
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image
#****************functions****************



def PreProcessing(img):
    
    original_height, original_width = img.shape[:2]
    #========================================
    img_width, img_height = 400,600
    small_img = cv.resize(img,(original_width,original_height))


    #========================================

    nobg_img= remove(small_img)

    #========================================

    imggray = cv.cvtColor(nobg_img,cv.COLOR_BGR2GRAY)

    #========================================


    ret, thresholded = cv.threshold(imggray, 0, 255, cv.THRESH_BINARY)


    #========================================


    small_img_copy = small_img.copy()

    contours = cv.findContours(thresholded,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
    # largest_contour = max(contours, key = cv.contourArea)
    largest_contour = sorted(contours, key = cv.contourArea, reverse=True)[0]

    cv.drawContours(small_img_copy, [largest_contour],-1,(0,0,255),2)
    # cv.imshow("small_img_copy", small_img_copy)
    # cv.waitKey(0)
    #========================================

    unOrdered_corners = GetCornerPoints(largest_contour)
    drawed_Img_Corners = DrawPointsOnImage(small_img_copy.copy(),unOrdered_corners,"unOrdered_corners")
    ordered_corners_clockwise = OrderCornerPointClockwise(unOrdered_corners)
    # DrawPointsOnImage(small_img_copy,ordered_corners_clockwise,"ordered_corners_clockwise")

    #========================================



    warrped_img = ApplyTopView(small_img,np.float32(ordered_corners_clockwise))

    return (small_img,nobg_img,imggray,thresholded,small_img_copy,drawed_Img_Corners,warrped_img)



def PreProcessing2(img):
    """_summary_

    Args:
        img (_type_): _description_

    Returns:
        no bg image : _description_
        ordered_corners_clockwise : to rotate this image
    """
    
    original_height, original_width = img.shape[:2]
    #========================================
    small_img = cv.resize(img,(original_width,original_height))


    #========================================

    nobg_img= remove(small_img)

    #========================================

    imggray = cv.cvtColor(nobg_img,cv.COLOR_BGR2GRAY)

    #========================================


    ret, thresholded = cv.threshold(imggray, 0, 255, cv.THRESH_BINARY)


    #========================================


    small_img_copy = small_img.copy()

    contours = cv.findContours(thresholded,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
    # largest_contour = max(contours, key = cv.contourArea)
    largest_contour = sorted(contours, key = cv.contourArea, reverse=True)[0]

    cv.drawContours(small_img_copy, [largest_contour],-1,(0,0,255),2)
    # cv.imshow("small_img_copy", small_img_copy)
    # cv.waitKey(0)
    #========================================

    unOrdered_corners = GetCornerPoints(largest_contour)
    drawed_Img_Corners = DrawPointsOnImage(small_img_copy.copy(),unOrdered_corners,"unOrdered_corners")
    ordered_corners_clockwise = OrderCornerPointClockwise(unOrdered_corners)
    # DrawPointsOnImage(small_img_copy,ordered_corners_clockwise,"ordered_corners_clockwise")

    #========================================




    return (ordered_corners_clockwise, small_img,nobg_img,imggray,thresholded,small_img_copy,drawed_Img_Corners)




def PreProcessing3(nobg_AngledImage):
    """_summary_

    Args:
        img (nobg_AngledImage): processed image which have no background, to crop it after rotation (first image processing)

    Returns:
       ordered_corners_clockwise
       thresholded
       small_img_copy
       drawed_Img_Corners
    """
    
  

    #========================================

    imggray = cv.cvtColor(nobg_AngledImage,cv.COLOR_BGR2GRAY)

    #========================================


    ret, thresholded = cv.threshold(imggray, 0, 255, cv.THRESH_BINARY)


    #========================================


    small_img_copy = nobg_AngledImage.copy()

    contours = cv.findContours(thresholded,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
    # largest_contour = max(contours, key = cv.contourArea)
    largest_contour = sorted(contours, key = cv.contourArea, reverse=True)[0]

    cv.drawContours(small_img_copy, [largest_contour],-1,(0,0,255),2)
    # cv.imshow("small_img_copy", small_img_copy)
    # cv.waitKey(0)
    #========================================

    unOrdered_corners = GetCornerPoints(largest_contour)
    drawed_Img_Corners = DrawPointsOnImage(small_img_copy.copy(),unOrdered_corners,"unOrdered_corners")
    ordered_corners_clockwise = OrderCornerPointClockwise(unOrdered_corners)
    # DrawPointsOnImage(small_img_copy,ordered_corners_clockwise,"ordered_corners_clockwise")

    #========================================




    return (ordered_corners_clockwise,thresholded,small_img_copy,drawed_Img_Corners)




if __name__ == "__main__":
    
    # URL of the image
    url = "http://www.akremplatform.somee.com//images//Branches//eb7c84b8-e926-44b4-afe9-0c7b4513acd4.jpg"

    # Fetch the image from the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the bytes data to a numpy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        
        # Decode the numpy array to an image
        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        
        # Display the image
        cv.imshow('Image from URL', image)
        cv.waitKey(0)  # Wait for a key press to close the image window
        cv.destroyAllWindows()
        
        
    
    # imagepath = r"http://www.akremplatform.somee.com//images//Branches//eb7c84b8-e926-44b4-afe9-0c7b4513acd4.jpg"
    # image = cv.imread(imagepath)
    # small_img,nobg_img,imggray,thresholded,small_img_copy,drawed_Img_Corners,warrped_img = PreProcessing(image)
    
    # fig, axes = plt.subplots(3,3)

    # axes[0,0].imshow(cv.cvtColor(small_img, cv.COLOR_BGR2RGB))
    # axes[0,0].set_title('Original')

    # axes[0,1].imshow(cv.cvtColor(nobg_img, cv.COLOR_BGR2RGB))
    # axes[0,1].set_title('No Background')

    # axes[0,2].imshow(cv.cvtColor(imggray, cv.COLOR_BGR2RGB))
    # axes[0,2].set_title('Gray Scale')

    # axes[1,0].imshow(cv.cvtColor(thresholded, cv.COLOR_BGR2RGB))
    # axes[1,0].set_title('Thresholded')

    # axes[1,1].imshow(cv.cvtColor(small_img_copy, cv.COLOR_BGR2RGB))
    # axes[1,1].set_title('Drawed Contours')
    
    # axes[1,2].imshow(cv.cvtColor(drawed_Img_Corners, cv.COLOR_BGR2RGB))
    # axes[1,2].set_title('Polygon Corners')
    
    # # axes[2,0].imshow(cv.cvtColor(warrped_img, cv.COLOR_BGR2RGB))
    # # axes[2,0].set_title('warrped image')

    



    # for ax in axes:
    #     for x in ax:
    #         x.axis('off')


    # plt.show()
    # plt.waitforbuttonpress(0)

