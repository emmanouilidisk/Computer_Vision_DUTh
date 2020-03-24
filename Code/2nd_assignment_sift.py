"""
2nd Assignment on Computer Vision

Author: Emmanouilidis Konstantinos
"""
 
import numpy as np
import cv2 as cv


def match(d1, d2):
"""
Function for finding matches between to images 
"""
 n1 = d1.shape[0]
 n2 = d2.shape[0]
 matches = []
 for i in range(n1):
  distances = np.sum(np.abs(d2 - d1[i, :]), axis=1)
  i2 = np.argmin(distances)
  mindist2 = distances[i2]
  distances[i2] = np.inf
  i3 = np.argmin(distances)
  mindist3 = distances[i3]
  if mindist2 / mindist3 < 0.5:
   matches.append(cv.DMatch(i, i2, mindist2))
 return matches

# Creating SIFT Object for later use
sift = cv.xfeatures2d_SIFT.create()

## Processing of first image ##
# Reading the file of the image
img1 = cv.imread('yard-03.png', cv.IMREAD_GRAYSCALE)
# Detecting keypoints and descriptors of first image
kp1 = sift.detect(img1)
desc1 = sift.compute(img1, kp1)

## Processing of second image ##
# Reading the file of the image
img2 = cv.imread('yard-02.png',cv.IMREAD_GRAYSCALE)
# Detecting keypoints and descriptors of second image
kp2 = sift.detect(img2)
desc2 = sift.compute(img2, kp2)

## Processing of third image ##
# Reading the file of the image
img3 = cv.imread('yard-01.png',cv.IMREAD_GRAYSCALE)
# Detecting keypoints and descriptors of third image
kp3 = sift.detect(img3)
desc3 = sift.compute(img3, kp3)

## Processing of fourth image ##
# Reading the file of the image
img4 = cv.imread('yard-00.png',cv.IMREAD_GRAYSCALE)
# Detecting keypoints and descriptors of fourth image
kp4 = sift.detect(img4)
desc4 = sift.compute(img4, kp4)

print("Starting ...")

##Joining first two images ##
#Finding matches between img1 and img2
matches1 = match(desc1[1], desc2[1])
img_pt1 = np.array([kp1[x.queryIdx].pt for x in matches1])
img_pt2 = np.array([kp2[x.trainIdx].pt for x in matches1])


# Finding homography
M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)

img5 = cv.warpPerspective(img2, M, (img1.shape[1]+1000, img1.shape[0]+1000))
img5[0: img2.shape[0], 0: img2.shape[1]] = img1



## Joining previous joined image with img3 ##
# Detecting keypoints and descriptors of image
kp5 = sift.detect(img5)
desc5 = sift.compute(img5, kp5)
# Finding matches between img5 and img3
matches2 = match(desc5[1], desc3[1])
img_pt5 = np.array([kp5[x.queryIdx].pt for x in matches2])
img_pt3 = np.array([kp3[x.trainIdx].pt for x in matches2])

# Finding homography
M, mask = cv.findHomography(img_pt3, img_pt5, cv.RANSAC)

# Wrap images appropriately
img6 = cv.warpPerspective(img3, M, (img5.shape[1]+1000, img5.shape[0]+1000))

img5_x=img5.shape[0]
img5_y=img5.shape[1]
img_black_white = np.ones((img5_x,img5_y), dtype=np.uint8)
for x in range(img5_x):
 for y in range(img5_y):
  if img5[x, y] == 0:
    img_black_white[x,y] = 0
strel = np.ones((7,7), np.uint8)
erode = cv.morphologyEx(img_black_white, cv.MORPH_ERODE, strel)
for x in range(erode.shape[0]):
 for y in range(erode.shape[1]):
  if erode[x, y] != 0:
   img6[x, y] = img5[x, y]


## Joining previous joined image with img4 ##
# Detecting keypoints and descriptors of image
kp6 = sift.detect(img6)
desc6 = sift.compute(img6, kp6)
# Finding matches
matches3 = match(desc6[1], desc4[1])
img_pt6 = np.array([kp6[x.queryIdx].pt for x in matches3])
img_pt4 = np.array([kp4[x.trainIdx].pt for x in matches3])
# Finding homography
M, mask = cv.findHomography(img_pt4, img_pt6, cv.RANSAC)

# Join img4 and img6 to get the final panorama img7
img7 = cv.warpPerspective(img4, M, (img6.shape[1]+1000, img6.shape[0]+1000))
kp7 = sift.detect(img7)
desc7 = sift.compute(img7, kp7)

img6_x=img6.shape[0]
img6_y=img6.shape[1]
img_black_white = np.ones((img6_x,img6_y), dtype=np.uint8)
for x in range(img6_x):
 for y in range(img6_y):
  if img6[x, y] == 0:
   img_black_white[x,y] = 0

## Erosing img_black_white to fill ##
strel = np.ones((11,11), np.uint8)
erode2 = cv.morphologyEx(img_black_white, cv.MORPH_ERODE, strel)
for x in range(erode2.shape[0]):
 for y in range(erode2.shape[1]):
  if erode2 [x, y] != 0:
    img7[x, y] = img6[x, y]


## Show images in different windows ##
# Show first image
cv.namedWindow('main1', cv.WINDOW_NORMAL)
cv.imshow('main1', img1)
cv.waitKey(0)
# Show second image
cv.namedWindow('main2', cv.WINDOW_NORMAL)
cv.imshow('main2', img2)
cv.waitKey(0)
# Show third image
cv.namedWindow('main3', cv.WINDOW_NORMAL)
cv.imshow('main3', img3)
cv.waitKey(0)
# Show fourth image
cv.namedWindow('main4', cv.WINDOW_NORMAL)
cv.imshow('main4', img4)
cv.waitKey(0)

cv.namedWindow('panorama', cv.WINDOW_NORMAL)
cv.imshow('panorama', img7)
cv.imwrite('panorama_sift.png', img7)
cv.waitKey(0)


"""
# <---------------------------------------------- SURF-----------------------------------------------> #

## Importing necessary libraries ##
import numpy as np
import cv2 as cv

## Function for finding matches between to images ##
def match(d1, d2):
 n1 = d1.shape[0]
 n2 = d2.shape[0]
 matches = []
 for i in range(n1):
  distances = np.sum(np.abs(d2 - d1[i, :]), axis=1)
  i2 = np.argmin(distances)
  mindist2 = distances[i2]
  distances[i2] = np.inf
  i3 = np.argmin(distances)
  mindist3 = distances[i3]
  if mindist2 / mindist3 < 0.5:
   matches.append(cv.DMatch(i, i2, mindist2))
 return matches

# Creating SURF Object for later use
surf = cv.xfeatures2d_SURF.create()

## Processing of first image ##
# Reading the file of the image
img1 = cv.imread('yard-03.png', cv.IMREAD_GRAYSCALE)
# Detecting keypoints and descriptors of first image
kp1 = surf.detect(img1)
desc1 = surf.compute(img1, kp1)

## Processing of second image ##
# Reading the file of the image
img2 = cv.imread('yard-02.png',cv.IMREAD_GRAYSCALE)
# Detecting keypoints and descriptors of second image
kp2 = surf.detect(img2)
desc2 = surf.compute(img2, kp2)

## Processing of third image ##
# Reading the file of the image
img3 = cv.imread('yard-01.png',cv.IMREAD_GRAYSCALE)
# Detecting keypoints and descriptors of third image
kp3 = surf.detect(img3)
desc3 = surf.compute(img3, kp3)

## Processing of fourth image ##
# Reading the file of the image
img4 = cv.imread('yard-00.png',cv.IMREAD_GRAYSCALE)
# Detecting keypoints and descriptors of fourth image
kp4 = surf.detect(img4)
desc4 = surf.compute(img4, kp4)

print("Starting ...")

## Joining first two images ##
# Finding matches between img1 and img2
matches1 = match(desc1[1], desc2[1])
img_pt1 = np.array([kp1[x.queryIdx].pt for x in matches1])
img_pt2 = np.array([kp2[x.trainIdx].pt for x in matches1])


# Finding homography
M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)

img5 = cv.warpPerspective(img2, M, (img1.shape[1]+1000, img1.shape[0]+1000))
img5[0: img2.shape[0], 0: img2.shape[1]] = img1


## Joining previous joined image with img3 ##
# Detecting keypoints and descriptors of image
kp5 = surf.detect(img5)
desc5 = surf.compute(img5, kp5)
# Finding matches between img5 and img3
matches2 = match(desc5[1], desc3[1])
img_pt5 = np.array([kp5[x.queryIdx].pt for x in matches2])
img_pt3 = np.array([kp3[x.trainIdx].pt for x in matches2])

# Finding homography
M, mask = cv.findHomography(img_pt3, img_pt5, cv.RANSAC)

# Wrap images approprietely
img6 = cv.warpPerspective(img3, M, (img5.shape[1]+1000, img5.shape[0]+1000))

'''
## Simple way to join the two images but not efficient due to black lines between images ##
for x in range (img5.shape[0]):
 for y in range (img5.shape[1]):
 if(img5[x,y]!=0):
 img6[x,y]=img5[x,y]
'''

img5_x=img5.shape[0]
img5_y=img5.shape[1]
img_black_white = np.ones((img5_x,img5_y), dtype=np.uint8)

for x in range(img5_x):
 for y in range(img5_y):
  if img5[x, y] == 0:
    img_black_white[x,y] = 0

strel = np.ones((7,7), np.uint8)
erode = cv.morphologyEx(img_black_white, cv.MORPH_ERODE, strel)
for x in range(erode.shape[0]):
 for y in range(erode.shape[1]):
  if erode[x, y] != 0:
   img6[x, y] = img5[x, y]


## Joining previous joined image with img4 ##
# Detecting keypoints and descriptors of image
kp6 = surf.detect(img6)
desc6 = surf.compute(img6, kp6)
# Finding matches
matches3 = match(desc6[1], desc4[1])
img_pt6 = np.array([kp6[x.queryIdx].pt for x in matches3])
img_pt4 = np.array([kp4[x.trainIdx].pt for x in matches3])
# Finding homography
M, mask = cv.findHomography(img_pt4, img_pt6, cv.RANSAC)

# Join img4 and img6 to get the final panorama img7
img7 = cv.warpPerspective(img4, M, (img6.shape[1]+1000, img6.shape[0]+1000))
kp7 = surf.detect(img7)
desc7 = surf.compute(img7, kp7)
'''
## Simple way to join the two images but not efficient due to black lines between images ##
for x in range (img6.shape[0]):
 for y in range (img6.shape[1]):
 if(img6[x,y]!=0):
   img7[x,y]=img6[x,y]
'''
img6_x=img6.shape[0]
img6_y=img6.shape[1]
img_black_white = np.ones((img6_x,img6_y), dtype=np.uint8)

for x in range(img6_x):
 for y in range(img6_y):
  if img6[x, y] == 0:
   img_black_white[x,y] = 0

## Erosing img_black_white to fill ##
strel = np.ones((11,11), np.uint8)
erode2 = cv.morphologyEx(img_black_white, cv.MORPH_ERODE, strel)
for x in range(erode2.shape[0]):
 for y in range(erode2.shape[1]):
  if erode2 [x, y] != 0:
    img7[x, y] = img6[x, y]


## Show images in different windows ##
# Show first image
cv.namedWindow('main1', cv.WINDOW_NORMAL)
cv.imshow('main1', img1)
cv.waitKey(0)
# Show second image
cv.namedWindow('main2', cv.WINDOW_NORMAL)
cv.imshow('main2', img2)
cv.waitKey(0)
# Show third image
cv.namedWindow('main3', cv.WINDOW_NORMAL)
cv.imshow('main3', img3) 
cv.waitKey(0)
# Show fourth image
cv.namedWindow('main4', cv.WINDOW_NORMAL)
cv.imshow('main4', img4)
cv.waitKey(0)

cv.namedWindow('panorama', cv.WINDOW_NORMAL)
cv.imshow('panorama', img7)
cv.imwrite('panorama_surf.png', img7)
cv.waitKey(0)
"""

