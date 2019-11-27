import pandas as pd 
import numpy as np 
from PIL import Image
import cv2
import pytesseract
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
TESSDATA_PREFIX = 'C:/Program Files/Tesseract-OCR'

statement = pytesseract.image_to_string(Image.open('statement2.png'),lang='eng',config='-c preserve_interword_spaces=1x1 --psm 1 --oem 3')
df = pytesseract.image_to_data(Image.open('statement2.png'),lang='eng', output_type='data.frame')


from tabula import read_pdf
df = read_pdf('statement_sample1.pdf')

print(df)
print(statement)

statement




import re



s = statement
start = s.find('\n\n \n\n \n\n') + 8
end = s.find('\n\nGross*', start)
text = s[start:end]

print(text)




# Program to show various ways to read and 
# write data in a file. 
file1 = open("myfile.txt","w") 

  
# \n is placed to indicate EOL (End of Line) 

file1.writelines(text) 
file1.close() #to change file access modes 
  
file1 = open("myfile.txt","r+")  
  

print (file1.read())


import csv

with open("myfile.txt", 'r+') as infile, open("mycsv.csv", 'w') as outfile:
     stripped = (line.strip() for line in infile)
     lines = (line.split("/s/s") for line in stripped if line)
     writer = csv.writer(outfile)
     writer.writerows(lines)

df
img = cv2.imread('statement2.png')

plt.figure(figsize = (20,20))
plt.imshow(img)
plt.show()

crp = img[700:1200,100:1100]

# Convert to gray
crp = cv2.cvtColor(crp, cv2.COLOR_BGR2GRAY)

# Apply dilation and erosion to remove some noise
kernel = np.ones((1, 1), np.uint8)
crp = cv2.dilate(crp, kernel, iterations=1)
crp = cv2.erode(crp, kernel, iterations=1)

#  Apply threshold to get image with only black and white
crp = cv2.adaptiveThreshold(crp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

plt.figure(figsize = (20,20))
plt.imshow(crp)
plt.show()


# Recognize text with tesseract for python
result = pytesseract.image_to_string(crp, lang='eng',config='-c preserve_interword_spaces=1x1 --psm 
1 --oem 3')

df = pytesseract.image_to_data(crp,lang='eng', output_type='data.frame')

print(df)

import numpy
df = []
table_list = result.split('\n')
df = pd.DataFrame(table_list)
df



import os
import cv2
import imutils

# This only works if there's only one table on a page
# Important parameters:
#  - morph_size
#  - min_text_height_limit
#  - max_text_height_limit
#  - cell_threshold
#  - min_columns


def pre_process_image(pre, save_in_file, morph_size=(8, 8)):

    # get rid of the color
    #pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    #pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # dilate the text to make it solid spot
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.erode(cpy, struct, anchor=(-1, -1), iterations=1)
    cpy = cv2.dilate(cpy, struct, anchor=(-1, -1), iterations=1)
    pre = ~cpy

    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre


def find_text_boxes(pre, min_text_height_limit=6, max_text_height_limit=40):
    # Looking for the text spots contours
    # OpenCV 3
    # img, contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Getting the texts bounding boxes based on the text size assumptions
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]

        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)

    return boxes


def find_table_in_boxes(boxes, cell_threshold=10, min_columns=2):
    rows = {}
    cols = {}

    # Clustering the bounding boxes by their positions
    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

    # Filtering out the clusters having less than 2 cols
    table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    # Sorting the row cells by x coord
    table_cells = [list(sorted(tb)) for tb in table_cells]
    # Sorting rows by the y coord
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

    return table_cells


def build_lines(table_cells):
    if table_cells is None or len(table_cells) <= 0:
        return [], []

    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

    hor_lines = []
    ver_lines = []

    for box in table_cells:
        x = box[0][0]
        y = box[0][1]
        hor_lines.append((x, y, max_x, y))

    for box in table_cells[0]:
        x = box[0]
        y = box[1]
        ver_lines.append((x, y, x, max_y))

    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x, max_y, max_x, max_y))

    return hor_lines, ver_lines




pre_file = os.path.join("pre.png")
out_file = os.path.join("out.png")



pre_processed = pre_process_image(crp, pre_file)
text_boxes = find_text_boxes(pre_processed)
cells = find_table_in_boxes(text_boxes)
hor_lines, ver_lines = build_lines(cells)

# Visualize the result
vis = crp.copy()

# for box in text_boxes:
#     (x, y, w, h) = box
#     cv2.rectangle(vis, (x, y), (x + w - 2, y + h - 2), (0, 255, 0), 1)

for line in hor_lines:
    [x1, y1, x2, y2] = line
    cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

for line in ver_lines:
    [x1, y1, x2, y2] = line
    cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imwrite(out_file, vis)





























