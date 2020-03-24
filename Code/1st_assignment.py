"
This is the 1st Assignment of Computer Vision course
at ECE DUTh by Emmanouilidis Konstantinos
"


import cv2           
import numpy as np   
import math          


def median_filter(img, filter_size):
    "Function for median_filtering
    
    Parameters:
    img: image to be filtered
    filter_size: the size of the kernel

    Returns: the filtered image
    "
    
    temp = []
    indexer = filter_size // 2
    window = [
        (i, j)
        for i in range(-indexer, filter_size-indexer)
        for j in range(-indexer, filter_size-indexer)
    ]
    index = len(window) // 2
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = sorted(
                0 if (
                    min(i+a, j+b) < 0
                    or len(img) <= i+a
                    or len(img[0]) <= j+b
                ) else img[i+a][j+b]
                for a, b in window
            )[index]
    return img

# Select filename: "N5" for image with noise, "NF5" for image without noise
filename = 'N5.png'
#filename = 'NF5.png'

# Read image
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
print(img.shape)  

# Show image
cv2.namedWindow('main', )
cv2.imshow('main', img)
cv2.waitKey(0)

# Denoising image with median filter
img = median_filter(img,5)

## Show image after median filter
# cv2.imshow('main2', img)
# cv2.waitKey(0)

# Convert to binary image
th, img_th = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)     #using 75 as threshold
#th, img_th2 = cv2.threshold(img, 65, 255, cv2.THRESH_BINARY);  #using 65 as threshold

## Show binary image
# cv2.namedWindow('calc', )
# cv2.imshow('calc', img_th)
# cv2.waitKey(0)


#Closing image for extracting a small point seemingly like a cell
strel = np.ones((5,5), np.uint8)
img_th = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, strel)

#Show binary image after closing
cv2.namedWindow('calc', )
cv2.imshow('calc', img_th )
cv2.waitKey(0)

##Erosion for separating two simingly joint cells
# strel = np.ones((5,5), np.uint8)
# img_th = cv2.morphologyEx(img_th, cv2.MORPH_ERODE, strel)
# cv2.namedWindow('calc', )
# cv2.imshow('calc', img_th)
# cv2.waitKey(0)
#

#Finding contours of the cells
img_temp,contours,hierarchy= cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#Removing the cells that are in the borders of the image
count = 0
to_be_deleted = []
print("Initial len of contours",len(contours))
for i in range(len(contours)):
    for j in range(len(contours[i])):
        if (contours[i][j][0][0] == 0  ):
                to_be_deleted.append(i)
                break
        if (contours[i][j][0][0] == 806 ):
            to_be_deleted.append(i)
            break
        if (contours[i][j][0][1] == 0):
                    to_be_deleted.append(i)
                    break
        if (contours[i][j][0][1] == 564):
            to_be_deleted.append(i)
            break

##Printing the indexes of cells that are in the borders and will be deleted
#print(to_be_deleted)

# Deleting cells that are in the border of the image
del contours[to_be_deleted[0]]
for i in range(1,len(to_be_deleted)):
    del contours[to_be_deleted[i]-i]

#Show binary image
cv2.namedWindow('calc', )
cv2.drawContours(img_th,contours, -1, 100, 2)
cv2.waitKey(0)

#Printing number of contours
print("Number of contours", len(contours))

#<------------------------------------------ Task 2 of Exercise --------------------------------------------------->#
# Measuring area of every cell with contourArea()

area = []
for i in range(len(contours)):
    area.append(math.ceil(cv2.contourArea(contours[i])))
print("Number of pixels in each cell",area)


#<-------------------------------------- Task 3 of Exercise -------------------------------------------------------->#

# Declaration of lists used to save the coordinates of bounding boxes
x = []
y = []
w = []
h = []

# Finding the bounding boxes
for i in range(len(contours)):
    temp1,temp2,temp3,temp4 = cv2.boundingRect(np.asarray(contours[i]))
    x.append(temp1)
    y.append(temp2)
    w.append(temp3)
    h.append(temp4)
    cv2.rectangle(img_th,(x[i],y[i]),(x[i]+w[i],y[i]+h[i]),120,2)

cv2.imshow('calc',img_th)
cv2.waitKey(0)

# Creating the integral image and saving it in integral_list
integral_list = cv2.integral(img_th)

# Declaration of lists used in computing the mean value of grayscale pixels
A = []
B = []
C = []
D = []
gray_mean_value = []

# Computing the mean value of grayscale pixels
for i in range(len(contours)):
    A.append( integral_list[y[i]][x[i]] )
    D.append(integral_list[y[i]+h[i]][x[i]+w[i]])
    B.append(integral_list[y[i]][x[i]+w[i]])
    C.append(integral_list[ y[i]+h[i]][x[i]])

    #Division with number of pixels in bounding box
    gray_mean_value.append((A[i] + D[i] - B[i] - C[i]) / (w[i] * h[i]))

# Printing the mean value
print("Mean value of grayscale in each box", gray_mean_value)


