import sys
import numpy as np
import cv2
import skimage.filters


def updateScreenCnt(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # find closest point
        minDiffIdx = None
        minDiff = sys.maxsize
        for i in range(4):
            x0 = paperPoints[i, 0, 0]
            y0 = paperPoints[i, 0, 1]
            diff = (x0 - x) ** 2 + (y0 - y) ** 2
            if diff < minDiff:
                minDiffIdx = i
                minDiff = diff

        paperPoints[minDiffIdx, 0, 0] = x
        paperPoints[minDiffIdx, 0, 1] = y
        print("You have updated point {}.".format(minDiffIdx))


def transformOfRectangle(image, pts):
    # order points
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    # find new points
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # make transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# Load the image #
imageAddress = 'images/page.jpg'
image = cv2.imread(imageAddress)
desiredH = 700
ratio = image.shape[0] / desiredH
orig = image.copy()

desiredW = int(image.shape[1] * desiredH / image.shape[0])
image = cv2.resize(image, (desiredW, desiredH),
                   interpolation=cv2.INTER_AREA)

# Pre processing & edge detection #
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# cv2.imshow("Image", image)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Find rect #
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1]  # OpenCV v3, v4-pre, or v4-alpha
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
paperPoints = None
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:  # rect
        paperPoints = approx
        break
print("Please edit the corners of the paper if"
      " needed by click on the pic.")
print("When you feel ok with the corners press s.")
originalIMG = image.copy()
cv2.namedWindow("Contour")
cv2.setMouseCallback("Contour", updateScreenCnt)

while True:
    cv2.drawContours(image, [paperPoints], -1, (0, 0, 255), 3)
    cv2.imshow("Contour", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):  # scan
        break
    image = originalIMG.copy()

cv2.destroyAllWindows()

# Perspective transform #
warped = transformOfRectangle(orig, paperPoints.reshape(4, 2) * ratio)

# Post processing #
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = skimage.filters.threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

desiredH = 700
desiredW = int(orig.shape[1] * desiredH / orig.shape[0])
orig = cv2.resize(orig, (desiredW, desiredH),
                  interpolation=cv2.INTER_AREA)

desiredW = int(warped.shape[1] * desiredH / warped.shape[0])
warped = cv2.resize(warped, (desiredW, desiredH),
                    interpolation=cv2.INTER_AREA)

cv2.imshow("Original", orig)
cv2.imshow("Scanned", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
