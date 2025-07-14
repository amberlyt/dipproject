# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 16:10:14 2025

@author: Chan Xian Kang
"""

import cv2
import numpy as np
import matplotlib.pyplot as pt

'''
* Read grayscale image, convert it to binary 
* Return the paragraphs stored in a list @paragraphs
* param   @filename  :  filename of the image
'''
def separatePara(filename):
    # Step 1: Read grayscale image
    img = cv2.imread(filename, 0)
    # Step 2: Binary conversion 
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Step 3: Dilation to group lines into paragraphs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 25))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    # Step 4: Find contours from dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paragraphs = [] # paragraph list store paragraphs 
    # Step 5: Sort paragraphs top-to-bottom by Y position
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 200 and h > 40:  # Tighter filter to reduce noise
            paragraph = img[y:y+h, x:x+w]
            paragraphs.append(paragraph)
            bounding_boxes.append((x, y, w, h))
    # Sort by y (top-to-bottom)
    paragraphs = [para for _, para in sorted(zip(bounding_boxes, paragraphs), key=lambda pair: pair[0][1])]
    return paragraphs    


'''
* display the paragraphs in a (single) image
* param   @paraList  :  list of the paragraph (returned from separatePara)
'''
def displayImage(paraList):
    print("Total paragraphs detected:", len(paraList))  #remove if dont want
    for i, para in enumerate(paraList):
        cv2.imshow(f"Paragraph {i+1}", para)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/001.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/002.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/003.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/004.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/005.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/006.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/007.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/008.png"))





