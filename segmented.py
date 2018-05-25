from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np

DIR = "/root/gcloudstuff/"

# define the dictionary of the 50 segments
DIGITS_LOOKUP = {
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): 'Null',
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1): 'Full',
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): '-',
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1): '0',
    (0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1): '1',
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1): '2',
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1): '3',
    (1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1): '4',
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1): '5',
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1): '6',
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1): '7',
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1): '8',
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1): '9'
}

image = cv2.imread(DIR+"testing2.jpg")

image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

sh = image.shape
row = sh[0]
col = sh[1]
#print sh, row, col
c = np.array([[[0, 0]], [[0, col]], [[row, 0]], [[row, col]]])
print c


#warped = four_point_transform(gray, c.reshape(4, 2))

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#cv2.imwrite(DIR+"preprocess/segre1.jpg", thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imwrite(DIR+"preprocess/segre.jpg", thresh)

digits = []
for i in range(1):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = thresh[y:y + h, x:x + w]

    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.20), int(roiH * 0.10))
    dHC = int(roiH * 0.05)

    # define the 50 segements
    segments = [
	((0, 0), (dW, dH)), ((dW, 0), (dW*2, dH)), ((dW*2, 0), (dW*3, dH)), ((dW*3, 0), (dW*4, dH)), ((dW*4, 0), (dW*5, dH)), # row 1
	((0, dH), (dW, dH*2)), ((dW, dH), (dW*2, dH*2)), ((dW*2, dH), (dW*3, dH*2)), ((dW*3, dH), (dW*4, dH*2)), ((dW*4, dH), (dW*5, dH*2)), # row 2
	((0, dH*2), (dW, dH*3)), ((dW, dH*2), (dW*2, dH*3)), ((dW*2, dH*2), (dW*3, dH*3)), ((dW*3, dH*2), (dW*4, dH*3)), ((dW*4, dH*2), (dW*5, dH*3)), # row 3
	((0, dH*3), (dW, dH*4)), ((dW, dH*3), (dW*2, dH*4)), ((dW*2, dH*3), (dW*3, dH*4)), ((dW*3, dH*3), (dW*4, dH*4)), ((dW*4, dH*3), (dW*5, dH*4)), # row 4
	((0, dH*4), (dW, dH*5)), ((dW, dH*4), (dW*2, dH*5)), ((dW*2, dH*4), (dW*3, dH*5)), ((dW*3, dH*4), (dW*4, dH*5)), ((dW*4, dH*4), (dW*5, dH*5)), # row 5
	((0, dH*5), (dW, dH*6)), ((dW, dH*5), (dW*2, dH*6)), ((dW*2, dH*5), (dW*3, dH*6)), ((dW*3, dH*5), (dW*4, dH*6)), ((dW*4, dH*5), (dW*5, dH*6)), # row 6
	((0, dH*6), (dW, dH*7)), ((dW, dH*6), (dW*2, dH*7)), ((dW*2, dH*6), (dW*3, dH*7)), ((dW*3, dH*6), (dW*4, dH*7)), ((dW*4, dH*6), (dW*5, dH*7)), # row 7
	((0, dH*7), (dW, dH*8)), ((dW, dH*7), (dW*2, dH*8)), ((dW*2, dH*7), (dW*3, dH*8)), ((dW*3, dH*7), (dW*4, dH*8)), ((dW*4, dH*7), (dW*5, dH*8)), # row 8
	((0, dH*8), (dW, dH*9)), ((dW, dH*8), (dW*2, dH*9)), ((dW*2, dH*8), (dW*3, dH*9)), ((dW*3, dH*8), (dW*4, dH*9)), ((dW*4, dH*8), (dW*5, dH*9)), # row 9
	((0, dH*9), (dW, dH*10)), ((dW, dH*9), (dW*2, dH*10)), ((dW*2, dH*9), (dW*3, dH*10)), ((dW*3, dH*9), (dW*4, dH*10)), ((dW*4, dH*9), (dW*5, dH*10)) # row 10
    ]
    on = [0] * len(segments)

    # loop over the segments
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
	# extract the segment ROI, count the total number of
	# thresholded pixels in the segment, and then compute
	# the area of the segment
	segROI = roi[yA:yB, xA:xB]
	total = cv2.countNonZero(segROI)
	area = (xB - xA) * (yB - yA)

	# if the total number of non-zero pixels is greater than
	# 50% of the area, mark the segment as "on"
	if total / float(area) > 0.5:
	    on[i]= 1

    digit = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit)
    #cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #cv2.putText(output, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

print digits
for e in digits:
    print e
