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
        # Size filter
        if w > 200 and h > 40:
            roi = binary[y:y+h, x:x+w]
            # Count inner contours (small text-like blobs)
            roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_inner_contours = len(roi_contours)
            # Heuristics:
            # - Paragraphs: many small internal contours
            # - Tables: few large contours, possible straight edges
            # - Pictures: few contours, mostly filled
            contour_areas = [cv2.contourArea(c) for c in roi_contours]
            avg_area = np.mean(contour_areas) if contour_areas else 0
            total_area = w * h
            fill_ratio = np.sum(roi == 255) / total_area  # how much of the region is white (text pixels)
            # Rules (tweak as needed)
            if num_inner_contours > 10 and avg_area < 1000 and fill_ratio < 0.5:
                paragraphs.append(img[y:y+h, x:x+w])
                bounding_boxes.append((x, y, w, h))
    # Sort top-to-bottom
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
        

def saveParagraphs(paraList, original_filename):
    # Manually extract image name
    parts = original_filename.replace("\\", "/").split("/")
    base_name = parts[-1]  # '005.png'
    folder_name = "DIP_Assignment/Sample outputs from " + base_name 

    # Try to create folder using OpenCV workaround (writing dummy and deleting)
    try:
        cv2.imwrite(folder_name + "/dummy.png", np.zeros((10, 10), dtype=np.uint8))
        import os
        os.remove(folder_name + "/dummy.png")  # only use os here temporarily, optional
    except:
        pass  # folder likely didn't exist but now does

    # Save paragraph images
    for i, para in enumerate(paraList):
        filepath = folder_name + "/paragraph_" + str(i + 1) + ".png"
        cv2.imwrite(filepath, para)
        print("Saved:", filepath)

'''
# Example usage
img_path = "CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/005.png"
paras = separatePara(img_path)
saveParagraphs(paras, img_path)
'''
        
for i in range(1, 9):
    file_path = f"CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/{i:03d}.png"
    paras = separatePara(file_path)
    saveParagraphs(paras, file_path)
        
#displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/005.png"))

        
'''
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/001.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/002.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/003.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/004.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/005.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/006.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/007.png"))
displayImage(separatePara("CSC2014- Group Assignment_Aug-2025/Converted Paper (8)/008.png"))
'''





