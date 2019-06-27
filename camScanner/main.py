import numpy as np
import cv2
import skimage.filters


# def transformOfRectangle(image, pts):
#     # order points
#     rect = np.zeros((4, 2), dtype="float32")
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#
#     (tl, tr, br, bl) = rect
#
#     # find new points
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))
#
#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))
#
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")
#
#     # make transform
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#     return warped
#
#
# # Load the image
# imageAddress = 'images/page.jpg'
# image = cv2.imread(imageAddress)
# desiredH = 500
# ratio = image.shape[0] / desiredH
# orig = image.copy()
#
# desiredW = int(image.shape[1] * desiredH / image.shape[0])
# image = cv2.resize(image, (desiredW, desiredH),
#                    interpolation=cv2.INTER_AREA)
#
# # Pre processing & edge detection
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(gray, 75, 200)
#
# # cv2.imshow("Image", image)
# # cv2.imshow("Edged", edged)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # Find rect
# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[1]  # OpenCV v3, v4-pre, or v4-alpha
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
# screenCnt = None
# for c in cnts:
#     # approximate the contour
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#
#     if len(approx) == 4:  # rect
#         screenCnt = approx
#         break
# cv2.drawContours(image, [screenCnt], -1, (255, 0, 0), 1)
# # cv2.imshow("Outline", image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # Perspective transform
# warped = transformOfRectangle(orig, screenCnt.reshape(4, 2) * ratio)
#
# # Post processing
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = skimage.filters.threshold_local(warped, 11, offset=10, method="gaussian")
# warped = (warped > T).astype("uint8") * 255
#
# desiredH = 650
# desiredW = int(orig.shape[1] * desiredH / orig.shape[0])
# orig = cv2.resize(orig, (desiredW, desiredH),
#                   interpolation=cv2.INTER_AREA)
#
# desiredW = int(warped.shape[1] * desiredH / warped.shape[0])
# warped = cv2.resize(warped, (desiredW, desiredH),
#                     interpolation=cv2.INTER_AREA)
#
# cv2.imshow("Original", orig)
# cv2.imshow("Scanned", warped)
# cv2.waitKey(0)

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        print("LEFT BUTTON DOWN")
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        print("LEFT BUTTON UP")
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread('images/receipt.jpg')
desiredH = 500
ratio = image.shape[0] / desiredH

desiredW = int(image.shape[1] * desiredH / image.shape[0])
image = cv2.resize(image, (desiredW, desiredH),
                   interpolation=cv2.INTER_AREA)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()