from paddleocr import PaddleOCR,draw_ocr
import numpy as np
import cv2 as cv
import re
import math

def rectangle_area(v1, v2, v3, v4):
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    x4, y4 = v4

    area = 0.5 * abs((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))
    return area


def getAvgLetterArea(WordTitleAndCoordinate):
    y= WordTitleAndCoordinate[1]

    area=rectangle_area(y[0],y[1],y[2],y[3])
    AvgLetterArea = area /len(WordTitleAndCoordinate[0])
    return AvgLetterArea

def biggest_WordOrStatmentArea(WordsTitleAndCoordinate):
    max_area=0
    largest_word_details =0
    
    CountOfWords=0
    
    for i in range(len(WordsTitleAndCoordinate)):

        Words = WordsTitleAndCoordinate[i][0].split()

               
        if len(Words)>4 :
            continue
            
        
        y=WordsTitleAndCoordinate[i][1]

        area=rectangle_area(y[0],y[1],y[2],y[3])
        # AvgLetterArea = area /len(WordsTitleAndCoordinate[i][0]) 
        # AvgWordArea = area /len(Words) 


        if max_area<area:

          max_area=area
          largest_word_details= WordsTitleAndCoordinate[i]

    return largest_word_details

def biggest_WordArea(WordsTitleAndCoordinate):
    max_area=0
    largest_word_details =0
    
    CountOfWords=0
    
    for i in range(len(WordsTitleAndCoordinate)):

        words = WordsTitleAndCoordinate[i][0].split()

        if len(words)<1:    
            continue
            
            
        sortedWords = sorted(words, key=lambda x : len(x))
        biggestWordInStatment = sortedWords[-1]
        lengthOfBigWord = len(biggestWordInStatment)
 
          
        
        coordinate=WordsTitleAndCoordinate[i][1]

        area=rectangle_area(coordinate[0],coordinate[1],coordinate[2],coordinate[3])
        AvgLetterArea = area /len(WordsTitleAndCoordinate[i][0]) 
        
        
        BiggestWordStatmentArea = lengthOfBigWord*AvgLetterArea # to get the area of one word in statment


        if max_area<BiggestWordStatmentArea:

          max_area=BiggestWordStatmentArea
          largest_word_details= WordsTitleAndCoordinate[i]

    return largest_word_details

def GetDeltaY(ArrayofPoints):
  dy=np.abs(ArrayofPoints[0][1]-ArrayofPoints[3][1])
  return dy

def GetDeltaX(ArrayofPoints):
  dx=np.abs(ArrayofPoints[0][0]-ArrayofPoints[1][0])
  return dx


def is_convertible_to_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def correct_zero_to_o(statement):
    # Define a regular expression pattern to match words with '0' instead of 'o'
    pattern = r'\b(?![0-9]+(mg|ml))([a-zA-Z]*[0][a-zA-Z]*)\b'
    
    # Use regular expression to find words with '0' instead of 'o' and correct them
    corrected_statement = re.sub(pattern, lambda match: match.group().replace('0', 'o'), statement)
    
    return corrected_statement


def separate_alphabetic_and_numerical(input_string):
    # Define a regular expression pattern to match alphabetic words and numerical numbers
    pattern = r'(\d+|[a-z0A-Z\D*]+)'
    
    # Use regular expression to find all matches
    matches = re.findall(pattern, input_string)
    
    # Separate alphabetic words and numerical numbers and join them with whitespace
    separated_string = ' '.join(matches)
    
    return separated_string


def separate_special_chars(input_string):
    # Define a regular expression pattern to match special characters, alphabetic characters, and numbers
    pattern = r'([^\w\s])|([a-zA-Z]+)|(\d+)'
    
    # Use regular expression to find all matches
    matches = re.findall(pattern, input_string)
    
    # Separate special characters, alphabetic characters, and numbers and join them with whitespace
    separated_string = ' '.join(['{}'.format(match[0] if match[0] else match[1] if match[1] else match[2]) for match in matches])
    
    return separated_string


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

def angle_between_angles(angle1, angle2):
    # Calculate the absolute difference between the angles
    diff = abs(angle1 - angle2) % 360

    # Choose the smaller angle between the two
    return min(diff, 360 - diff)



def replace_zeros_between_letters(input_str):
    result = ''
    for i in range(len(input_str)):
        if input_str[i] == '0':
            if i > 0 and i < len(input_str) - 1:
                if input_str[i-1].isalpha() and input_str[i+1].isalpha():
                    result += 'O'
                else:
                    result += input_str[i]
            else:
                result += input_str[i]
        else:
            result += input_str[i]
    return result

def remove_concentration(medicine_name):
    # Define patterns to match common concentration formats
    concentration_patterns = [
        r'\d+\s*mg',    # Matches digits followed by 'mg'
        r'\d+\s*ml',    # Matches digits followed by 'ml'
        r'\d+\s*%\s*',  # Matches digits followed by '%'
        r'\d+\s*mcg',   # Matches digits followed by 'mcg'
        r'\d+\s*g',     # Matches digits followed by 'g'
        r'\d+\s*unit',  # Matches digits followed by 'unit'
        r'\d+\.\d+%$',  # Matches fraction digits followed by '%', e.g., 0.05%
        r'\d+/\d+\s*mg', # Matches fraction followed by 'mg', e.g., 16/50mg
        r'\d+/\d+\s*ml', # Matches fraction followed by 'mg', e.g., 16/50mg
    ]

    # Compile regex pattern
    concentration_regex = re.compile('|'.join(concentration_patterns), flags=re.IGNORECASE)

    # Remove concentration from medicine name
    cleaned_name = re.sub(concentration_regex, '', medicine_name)

    return cleaned_name.strip()  # Strip any leading or trailing whitespace

def remove_special_characters(text):
    # Define the regular expression pattern to match special characters
    pattern = r'[^a-zA-Z0-9\-_.&%/ ]'   
    # Use re.sub() to replace matched characters with an empty string
    cleaned_text = re.sub(pattern, '', text)   
    return cleaned_text


def remove_whitespace_around_special_chars(text):
    # Define the regex pattern to match whitespace adjacent to specific special characters
    pattern = r'\s*([.,/&\-_])\s*'
    # Substitute matched whitespace with the special character
    cleaned_text = re.sub(pattern, r'\1', text)
    return cleaned_text


def fix_word_Concentration(text):
    pattern=r"[0-9]+o+[0-9]*"
    corrected_concentration = re.sub(pattern, lambda match: match.group().replace('o', '0'), text)
    return corrected_concentration





def cleanMedicineName(name):
    
    name = replace_zeros_between_letters(name)
    name = (name.strip()).lower()
    name = fix_word_Concentration(name)
    # name = remove_concentration(name)
    name = remove_special_characters(name)
    name = remove_whitespace_around_special_chars(name)
    
    return name

#==================================================FILTERS===================================================
def contains_non_english_special_numeric(text):
    # Regular expression to match any character that is not English, special char, or numeric
    pattern = r'[^A-Za-z0-9\s!@#$%^&*()-_=+`~{}:;"\',./?]'

    # Check if the pattern matches any part of the text
    return bool(re.search(pattern, text))

def contains_sequence_of_E(text):
    # Regular expression to match a sequence of 'E' characters more than 2 times
    pattern = r'E{3,}'

    # Check if the pattern matches any part of the text
    return bool(re.search(pattern, text))
#==================================================FILTERS===================================================






def CreatePaddlePipline(useAngle = False):
   return PaddleOCR(rec_algorithm='CRNN',use_angle_cls=useAngle)


def PaddleOpticalCharacterRecognition(ImagePath, PaddlePipline, useAngle = False):
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    # to switch the language model in order.

    # one time call pipline

    # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    try:
        
        ocr = PaddlePipline # need to run only once to download and load model into memory

        result = ocr.ocr(ImagePath, cls=useAngle)

        if result[0] == None:
            return (0, "",0,"","","","",[0,0,0,0])

            
        

        allWordsTitle_Coordinate=[]
        allWordsScores=[]
        for i in range(len(result[0])):
            Word=result[0][i][1]
            WordTitle=Word[0].strip()
            WordScore=Word[1]
            WordCoordinate = result[0][i][0]
            
            #if length > width skip score calculation 
            #if length < width mean that : this word is alignmented so calculate its score ...
            if abs(WordCoordinate[3][1] - WordCoordinate[0][1])<abs(WordCoordinate[2][0] - WordCoordinate[3][0]):    
                allWordsScores.append(WordScore)      
            
            if contains_non_english_special_numeric(WordTitle):
                continue      
            
            if contains_sequence_of_E(WordTitle):
                continue      
            
            # ignore any word its accuracy less than 90 %
            if WordScore < 0.85 and not WordTitle.isalpha() and len(WordTitle)>1:
                continue
            
                   
            WordTitle=Word[0].strip()
            if (WordTitle) == 'EE':
                continue
            
            
            # WordTitle = separate_alphabetic_and_numerical(WordTitle)
            # WordTitle = correct_zero_to_o(WordTitle)
            # WordTitle = separate_special_chars(WordTitle)
            
                    
            for element  in allWordsTitle_Coordinate:
                cv.rectangle(ImagePath, (int(element[1][0][0]), int(element[1][0][1])), (int(element[1][2][0]), int(element[1][2][1])), (0, 255, 0), 2)
            
            allWordsTitle_Coordinate.append((WordTitle,WordCoordinate))  
            #==>

        scorePoints = sum(allWordsScores)

        x=allWordsTitle_Coordinate

        allWordsReturned = [wordData[0] for wordData in x]
        # print(allWordsReturned)
        largest_word_details = biggest_WordArea(allWordsTitle_Coordinate)
        PointsArray_of_largest_word = largest_word_details[1]






        words=[x[i][0] for i in range(len(x))]

    


        #*********************************** Get Name of medicine .....***********************************
        
        #for level 1 only...
        max_dy = GetDeltaY(PointsArray_of_largest_word)
        max_dx = GetDeltaX(PointsArray_of_largest_word)
        min_X =  (max_dx * (10/100))
        max_Y = np.round(max_dy) + (max_dy * (35/100))
        min_Y = np.round(max_dy) - (max_dy * (35/100))

        accepted_WordsTitle_Coordinate_Level1=[]
        
        array_Accebted_Words=[]
        for i in range(len(allWordsTitle_Coordinate)):
            array_ofPoints = allWordsTitle_Coordinate[i][1]
            if ( GetDeltaY(array_ofPoints) >= min_Y and  GetDeltaY(array_ofPoints) <= max_Y and GetDeltaX(array_ofPoints)>min_X):
                array_Accebted_Words.append(allWordsTitle_Coordinate[i][0])
                accepted_WordsTitle_Coordinate_Level1.append(allWordsTitle_Coordinate[i])

        #-----------------------------------------------------------
        
        
        
        #for level 2 and level 3
        max_dy = GetDeltaY(PointsArray_of_largest_word)
        max_dx = GetDeltaX(PointsArray_of_largest_word)
        min_X =  (max_dx * (10/100))
        max_Y = np.round(max_dy) + (max_dy * (25/100))
        min_Y = np.round(max_dy) - (max_dy * (25/100))

        accepted_WordsTitle_Coordinate_Level23=[]
        
        array_Accebted_Words=[]
        for i in range(len(allWordsTitle_Coordinate)):
            array_ofPoints = allWordsTitle_Coordinate[i][1]
            if ( GetDeltaY(array_ofPoints) >= min_Y and  GetDeltaY(array_ofPoints) <= max_Y and GetDeltaX(array_ofPoints)>min_X):
                array_Accebted_Words.append(allWordsTitle_Coordinate[i][0])
                accepted_WordsTitle_Coordinate_Level23.append(allWordsTitle_Coordinate[i])
        #-----------------------------------------------------------


        
        
        
        # max_LetterArea = getAvgLetterArea(largest_word_details)
        
        # min_LetterArea = (max_LetterArea * (50/100))
        
        # accepted_WordsTitle_Coordinate=[]
        # array_Accebted_Words=[]
        # for word in allWordsTitle_Coordinate:
        #     if getAvgLetterArea(word) >= min_LetterArea :
        #         array_Accebted_Words.append(word[0])
        #         accepted_WordsTitle_Coordinate.append(word)
                

        Text3Levels = [[],[],[]]

        #level one :- get 50% of MaxDeltaY
        y_axis_sortedWords1 = sorted(accepted_WordsTitle_Coordinate_Level1, key=lambda x: x[1][0][1])
        
        #level two and three :- get 20% of MaxDeltaY
        y_axis_sortedWords23 = sorted(accepted_WordsTitle_Coordinate_Level23, key=lambda x: x[1][0][1])

        
        for i in range(len(Text3Levels)):
            y_axis_sortedWords = []
            level = Text3Levels[i]
            
            if i == 0:
                #level one :- get 50% of MaxDeltaY
                y_axis_sortedWords = y_axis_sortedWords1
            else:
                #level two and three :- get 20% of MaxDeltaY
                y_axis_sortedWords = y_axis_sortedWords23

                   
            if len(y_axis_sortedWords)>0:
                level.append(y_axis_sortedWords[0])
                
                
                #you must remove it from both lists....
                
                deletedElement = y_axis_sortedWords[0]
                if deletedElement in y_axis_sortedWords1:
                    y_axis_sortedWords1.remove(deletedElement)
                    
                                       
                if deletedElement in y_axis_sortedWords23:
                    y_axis_sortedWords23.remove(deletedElement)
                
                
                
                
                for word in y_axis_sortedWords:
                    
                    
                    twoPointsForAngleDetection = [level[0],word]
                    twoPointsForAngleDetection = sorted(twoPointsForAngleDetection, key=lambda x: x[1][0][0])
                    
                    p4FirstWord = twoPointsForAngleDetection[0][1][3]
                    p3FirstWord = twoPointsForAngleDetection[0][1][2]
                    
                    p4SecondWord = twoPointsForAngleDetection[1][1][3]

                    # angle between p4 and p3 in first word..
                    angle1 = angle_between_points(p4FirstWord,p3FirstWord)
                    
                    #  >>>>  bec .. p4 first word is neighbour point for p4 second word
                    # angle between p4 first word and p4 second word..
                    angle2 = angle_between_points(p4FirstWord,p4SecondWord)
                    
                    
                    p1FirstWord = twoPointsForAngleDetection[0][1][0]
                    p1SecondWord = twoPointsForAngleDetection[1][1][0]
                    
                    
                    
                        
                    condition1= p1FirstWord[1] >= p1SecondWord[1] and p1FirstWord[1] < p4SecondWord[1]
                    condition2= p4FirstWord[1] > p1SecondWord[1] and p4FirstWord[1] <= p4SecondWord[1]
                   
                    if angle_between_angles(angle1,angle2) <= 10 and (condition1 or condition2):
                        level.append(word)  
            
            for textWord in level:
                                
                     #you must remove it from both lists....
                if textWord in y_axis_sortedWords1:
                    y_axis_sortedWords1.remove(textWord)
                                 
                if textWord in y_axis_sortedWords23:
                    y_axis_sortedWords23.remove(textWord)
                        
                
                        
                            
                    
        # Rearrange order of elements according to --> x-axis     

        Text3Levels[0] = sorted(Text3Levels[0], key=lambda word: word[1][0][0])
        Text3Levels[1] = sorted(Text3Levels[1], key=lambda word: word[1][0][0])
        Text3Levels[2] = sorted(Text3Levels[2], key=lambda word: word[1][0][0])
                
                     
        MedicineName = [word[0] for word in Text3Levels[0] + Text3Levels[1]+ Text3Levels[2]]
                
                         
        ExtractedMedicineName = " ".join(MedicineName)
        CleanedMedicineName = cleanMedicineName(ExtractedMedicineName)
              
              
        #*********************************** Get Concentration of medicine .....***********************************
        #pass
        

       
        
           
        return (scorePoints, CleanedMedicineName,0,PointsArray_of_largest_word,words,array_Accebted_Words,allWordsTitle_Coordinate,largest_word_details[0])
    except Exception as e:
        print(e)      
        return (0, "",0,"","","","",[0,0,0,0])
        






def PaddleOCR_TextDetection(image, PaddlePipline):
    
    """_summary_

    function take 2 argument :
    1-image of medicine (-with background / -without background) 
    2-paddle pipline
    
    Returns:
        scorepoints: sumition of all accepted words (must word accuracy not less than 85%) must your ocr score point more than or equal it..
        Largest Word: which image will rotate according to it in first time....
        Largest Word coordinates: which image will rotate according to it in first time....
    """
    
    
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    # to switch the language model in order.

    # one time call pipline

    ocr = PaddlePipline # need to run only once to download and load model into memory

    result = ocr.ocr(image, cls=True)




    allWordsTitle_Coordinate=[]
    allWordsScores=[]
    
    
    for i in range(len(result[0])):
        
        Word=result[0][i][1]

        WordScore=Word[1]
        # ignore any word its accuracy less than 90 %
        if WordScore < 0.85:
            continue
        
        
        
        WordTitle=Word[0]
              
        WordCoordinate = result[0][i][0]
     
        allWordsTitle_Coordinate.append((WordTitle,WordCoordinate))  
        allWordsScores.append(WordScore)
       
        for element  in allWordsTitle_Coordinate:
            cv.rectangle(image, (int(element[1][0][0]), int(element[1][0][1])), (int(element[1][2][0]), int(element[1][2][1])), (0, 255, 0), 2)
    scorePoints = sum(allWordsScores)


    largest_word_details = biggest_WordArea(allWordsTitle_Coordinate)
    
    NameOfLargestWord = largest_word_details[0]
    PointsArray_of_largest_word = largest_word_details[1]

    return (scorePoints,image, PointsArray_of_largest_word,NameOfLargestWord)













if __name__ == "__main__":
    
    imagee = cv.imread(r"C:\Users\Islam\OneDrive\Desktop\RealVersionOfModel version two\WrongImage\TestImages\All-Vent\huawei p30 415.jpg")
    paddlepipline =CreatePaddlePipline(True)
    scorePoints,image, PointsArray_of_largest_word,NameOfLargestWord = PaddleOCR_TextDetection(imagee,paddlepipline)
    cv.imshow("ddd",image)
    cv.waitKey()
    
    
    # Test the function
    statement = "I l0ve c0ding with Pyth0n and 50 mg is a concentration flum0x 500mg hello 500mg00"
    corrected_statement = correct_zero_to_o(statement)
    print(corrected_statement)