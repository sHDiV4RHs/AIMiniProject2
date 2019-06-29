from __future__ import print_function
import sys
import numpy as np
import cv2
# from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


# based on
# https://docs.opencv.org/3.4.3/d8/d83/tutorial_py_grabcut.html
# https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html

def grabCut(img):
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (0, 0, img.shape[0], img.shape[1])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img


def testGrabCutOnImage(fileName):
    ### Test grabCut on image ###
    img = cv2.imread(fileName)
    # img = cv2.resize(img, (128 * 4, 72 * 4))
    img = grabCut(img)
    cv2.imshow('outputOfGrabCutIMG', img)
    cv2.waitKey()
    cv2.imwrite('outputOfGrabCutIMG.jpg', img)


def testGrabCutOnWebcam():
    ### Test grabCut on webcam ###
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (128 * 4, 72 * 4))
            originalIMG = img.copy()
            img = grabCut(img)

            cv2.imshow('GRABCut Output', img)
            cv2.imshow('Original IMG', originalIMG)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    cap.release()

    cv2.destroyAllWindows()


def testGrabCutOnVideo(fileName):
    ### Test grabCut on video ###
    cap = cv2.VideoCapture(fileName)

    if not cap.isOpened():
        print("Unable to read the file.")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('outputGrabcut.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if ret:
            # img = cv2.resize(frame, (frame_width, frame_height))
            img = grabCut(frame)
            out.write(img)
            cv2.imshow('AI PROJECT :)', img)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    cap.release()
    out.release()

    cv2.destroyAllWindows()


def backGrounSubtractor(fileName):
    ## [create]
    backSubKNN = cv2.createBackgroundSubtractorKNN()
    backSubMOG2 = cv2.createBackgroundSubtractorMOG2()

    ## [capture]
    capture = cv2.VideoCapture(fileName)
    if not capture.isOpened:
        print('Unable to open camera or file')
        exit(0)

    if fileName == 0:
        desiredWidth = 128 * 4
        desiredHeight = 72 * 4
    else:
        desiredWidth = int(capture.get(3))
        desiredHeight = int(capture.get(4))

    bgIMG = cv2.imread('bg.jpg')
    bgIMG = bgIMG[300:300 + desiredHeight, 300:300 + desiredWidth]
    # cv2.imshow("cropped", bgIMG)
    # cv2.waitKey(0)

    if fileName != 0:
        outKNNBG = cv2.VideoWriter('outputKNNBG.avi',
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                   (desiredWidth, desiredHeight))
        outMOG2BG = cv2.VideoWriter('outputMOG2BG.avi',
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                    (desiredWidth, desiredHeight))
        outKNNGray = cv2.VideoWriter('outputKNNGray.avi',
                                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                     (desiredWidth, desiredHeight))
        outMOG2Gray = cv2.VideoWriter('outputMOG2Gray.avi',
                                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                      (desiredWidth, desiredHeight))
        print("In Progress..")
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        frame = cv2.resize(frame, (desiredWidth, desiredHeight))
        ## [apply]
        fgMaskKNN = backSubKNN.apply(frame)
        fgMaskMOG2 = backSubMOG2.apply(frame)

        originalIMG = frame
        grayIMG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = []
        for fgMask in [fgMaskKNN, fgMaskMOG2]:
            # canvas.append(fgMask)
            mask = np.where((fgMask == 0), 0, 1).astype('float_')

            # blurMask = cv2.GaussianBlur(mask, (15, 15), 0)
            kernelSize = 10
            kernel = np.ones((kernelSize, kernelSize), np.float32) / (kernelSize ** 2)
            blurMask = cv2.filter2D(mask, -1, kernel)
            # canvas.append(blurMask)

            finalGrayIMG = ((grayIMG * (1 - blurMask))[:, :, np.newaxis]
                            + originalIMG * blurMask[:, :, np.newaxis]).astype('uint8')
            canvas.append(finalGrayIMG)
            finalChangedBG = (bgIMG * (1 - blurMask[:, :, np.newaxis])
                              + originalIMG * blurMask[:, :, np.newaxis]).astype('uint8')
            canvas.append(finalChangedBG)

        ## [show]
        # show the current frame and the fg masks
        if fileName == 0:
            cv2.imshow('original', originalIMG)
            cv2.imshow('KNN Gray', canvas[0])
            cv2.imshow('KNN BG', canvas[1])
            cv2.imshow('MOG2 Gray', canvas[2])
            cv2.imshow('MOG2 BG', canvas[3])

            # cv2.imshow('original', originalIMG)
            # cv2.imshow('FG Mask', fgMask)
            # cv2.imshow('BLUR Mask', blurMask)
            # cv2.imshow('IMG Final Gray', finalGrayIMG)
            # cv2.imshow('IMG Final Change Background', finalChangedBG)
        else:
            outKNNGray.write(canvas[0])
            outKNNBG.write(canvas[1])
            outMOG2Gray.write(canvas[2])
            outMOG2BG.write(canvas[3])

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


# testGrabCutOnImage('hand.jpg')
# testGrabCutOnWebcam()
# testGrabCutOnVideo('resizedHand.avi')

backGrounSubtractor(0)
# backGrounSubtractor('resizedHand.avi')
