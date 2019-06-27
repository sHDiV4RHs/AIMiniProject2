from __future__ import print_function
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse


# def grabCut(img):
#     mask = np.zeros(img.shape[:2], np.uint8)
#
#     bgdModel = np.zeros((1, 65), np.float64)
#     fgdModel = np.zeros((1, 65), np.float64)
#
#     rect = (0, 0, img.shape[0], img.shape[1])
#     cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
#
#     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#     img = img * mask2[:, :, np.newaxis]
#     return img
#
#
# def testGrabCutOnImage():
#     ### Test grabCut on image ###
#     img = cv2.imread('myHand1.jpg')
#     img = cv2.resize(img, (128 * 4, 72 * 4))
#     img = grabCut(img)
#     cv2.imshow('Photo', img)
#     cv2.waitKey()
#     myImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.show()
#
#
# def testGrabCutOnWebcam():
#     ### Test grabCut on webcam ###
#     cap = cv2.VideoCapture(0)
#
#     if not cap.isOpened():
#         print("Error opening video stream or file")
#
#     while cap.isOpened():
#         ret, img = cap.read()
#         if ret:
#             img = cv2.resize(img, (128 * 4, 72 * 4))
#             img = grabCut(img)
#
#             cv2.imshow('AI PROJECT :)', img)
#
#             # Press Q on keyboard to  exit
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break
#
#         # Break the loop
#         else:
#             break
#
#     cap.release()
#
#     cv2.destroyAllWindows()
#
#
# def testGrabCutOnVideo():
#     ### Test grabCut on video ###
#     cap = cv2.VideoCapture('resizedHand.avi')
#
#     if not cap.isOpened():
#         print("Unable to read the file.")
#
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#
#     out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
#
#     while True:
#         ret, frame = cap.read()
#
#         if ret:
#             # img = cv2.resize(frame, (frame_width, frame_height))
#             img = grabCut(frame)
#             out.write(img)
#             cv2.imshow('AI PROJECT :)', img)
#
#             # Press Q on keyboard to stop recording
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         # Break the loop
#         else:
#             break
#
#     cap.release()
#     out.release()
#
#     cv2.destroyAllWindows()
#
#
# # testGrabCutOnImage()
# # testGrabCutOnWebcam()
# # testGrabCutOnVideo()


def backGrounSubtractor(fileName):
    ## [create]
    # create Background Subtractor objects
    backSubKNN = cv2.createBackgroundSubtractorKNN()
    backSubMOG2 = cv2.createBackgroundSubtractorMOG2()

    ## [capture]
    capture = cv2.VideoCapture(fileName)
    if not capture.isOpened:
        print('Unable to open camera or file')
        exit(0)

    if fileName != 0:
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))

        outKNN = cv2.VideoWriter('outputKNN.avi',
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                 (frame_width, frame_height))
        outMOG2 = cv2.VideoWriter('outputMOG2.avi',
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                  (frame_width, frame_height))

    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        if fileName == 0:
            frame = cv2.resize(frame, (128 * 4, 72 * 4))
        ## [apply]
        # update the background model
        fgMaskKNN = backSubKNN.apply(frame)
        fgMaskMOG2 = backSubMOG2.apply(frame)

        ## [display_frame_number]
        # get the frame number and write it on the current frame
        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        ## [display_frame_number]

        ## [show]
        # show the current frame and the fg masks
        if fileName == 0:
            cv2.imshow('Frame', frame)
            cv2.imshow('FG Mask KNN', fgMaskKNN)
            cv2.imshow('FG Mask MOG2', fgMaskMOG2)
        else:
            # print(frame_width, frame_height)
            # print(fgMaskKNN.shape)
            # print(frame.shape)
            # outKNN.write(frame)
            # outMOG2.write(frame)
            # outKNN.write(fgMaskKNN)
            # outMOG2.write(fgMaskMOG2)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


# backGrounSubtractor(0)
backGrounSubtractor('resizedHand.avi')