#!/usr/bin/env python
by: llpassarelli@gmail.com
'''
Keyboard shortcuts:
   ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np
import sys
import os
import glob
import math
import time
import datetime


if __name__ == '__main__':
    import sys
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 1
    try:
        cap = cv2.VideoCapture(1)
    except:
        print("capture error")

    def processing(im):
        # mask
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        r, g, b = cv2.split(im)
        (thresh, bb) = cv2.threshold(b, 110, 255, cv2.THRESH_BINARY)
        cv2.imshow('b', bb)
        mask = cleanup(bb)
        # defects
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([255, 255, 120])
        maskb = cv2.inRange(hsv, lower, upper)
        maskc = cv2.bitwise_not(maskb)
        maskb = cv2.bitwise_and(maskc, mask, mask=mask)
        res = cv2.bitwise_and(gray, gray, mask=maskb)
        res = cv2.equalizeHist(res)
        cv2.imshow('pre', res)
        return res

    def cleanup(im):
        kernel = np.ones((3, 3), np.uint8)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        imr = cv2.resize(
            im,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_CUBIC)
        op = cv2.morphologyEx(imr, cv2.MORPH_OPEN, kernel1)
        cl = cv2.morphologyEx(op, cv2.MORPH_CLOSE, kernel1)
        blur = cv2.GaussianBlur(cl, (5, 5), 3)
        mask = cv2.resize(
            blur,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC)
        # cv2.imshow("cl", mask)
        mask = cv2.bitwise_and(im, mask, mask=mask)
        return mask

    def inspect(imd):
        # cv2.imshow("contamination", imd)
        defects = []
        shapes = []
        contaminations = []
        mask = np.zeros((480, 640), np.uint8)
        # Find the contours
        try:
            _, contours, hierarchy = cv2.findContours(
                imd.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is not None:
                # get the actual inner list of hierarchy descriptions
                hierarchy = hierarchy[0]
                # For each contour, find the bounding rectangle and draw it
                for component in zip(contours, hierarchy):
                    currentContour = component[0]
                    currentHierarchy = component[1]
                    cnt_len = cv2.arcLength(currentContour, True)
                    currentContour = cv2.approxPolyDP(
                        currentContour, 0.0036 * cnt_len, True)
                    cnt_area = cv2.contourArea(currentContour)

                    # shape outermost parent components
                    if currentHierarchy[3] < 0 and len(
                            currentContour) > 6 and cnt_area > 9000 and cnt_area < 90000 and cv2.isContourConvex(currentContour):
                        shapes.append(currentContour)
                        cv2.drawContours(mask, [currentContour], 0, 255, -1)
                        #print("shape area:", cnt_area)

                    # defects outermost parent
                    elif cnt_area > 9000 and cnt_area < 90000 and currentHierarchy[3] < 0:
                        defects.append(currentContour)

                    # contamination innermost parent
                    elif cnt_len > 4 and currentHierarchy[2] < 0:
                        # add contamination in shapes not in defects
                        #(x,y),radius = cv2.minEnclosingCircle(currentContour)
                        x, y, w, h = cv2.boundingRect(currentContour)
                        centerA = (int(x), int(y))
                        for crc in shapes:
                            (x, y), radius = cv2.minEnclosingCircle(crc)
                            centerB = (int(x), int(y))
                            radius = int(radius)
                            dist = cv2.norm(centerA, centerB)
                            if dist < radius:
                                # contaminations.append(currentContour)
                                print(" ")

                                # print "\n"
                                # cv2.imshow("mask",mask)
        except:
            pass
        return shapes, defects, contaminations
    count = 0
    last_inspect = 0
    while True:
        e1 = cv2.getTickCount()

        ret, frame = cap.read()
        cnt_style = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.resize(frame, (640, 480))
        total_shapes = 0
        total_defects = 0

        '''MEAN TRIGGER - DETECTA O OBJETO NO CENTRO DA FRAME'''
        #frame[240:310, 250:320] = 255
        roi = frame[245:315, 250:320]
        mean = cv2.mean(roi)
        trigger = frame.copy()
        trigger[235:275, 190:430] = 0
        cv2.imshow('trigger', trigger)
        print('mean:', mean)
        #cv2.imshow('mean', frame)
        #
        '''prevent multiples inspections by delay'''
        t = -1 * (last_inspect - e1) / cv2.getTickFrequency()
        print('last time:', t)

        if mean[0] > 100 and t > 0.36:

            # PRE PROCESSING
            pre = processing(frame)
            # INSPECTION
            shapes, defects, contaminations = inspect(pre)

            # RESULT
            total_shapes = len(shapes)
            total_defects = len(defects)
            total_contaminations = len(contaminations)
            if total_shapes > 0 or total_defects > 0:
                # DRAW MASKS SHAPES E DEFECTS
                mask = np.zeros((480, 640), np.uint8)
                mask1 = mask.copy()
                mask2 = mask.copy()

                if total_shapes > 0:
                    for crc in shapes:
                        cv2.drawContours(mask1, [crc], 0, 255, -1)
                if total_defects > 0:
                    for dfc in defects:
                        cv2.drawContours(mask2, [dfc], 0, 255, -1)

                mask = cv2.bitwise_or(mask1, mask2)
                frame = cv2.bitwise_and(frame, frame, mask=mask)

                cv2.drawContours(frame, defects, -1, (0, 0, 250), cnt_style)
                # cv2.drawContours( frame, shapes, -1, (0, 250, 0), cnt_style )
                # cv2.drawContours( frame, contaminations, -1, (0, 0, 250), cnt_style )
                # DRAW POSITIONS
                colors = {}
                colors['red'] = (111, 104, 187)
                colors['blue'] = (176, 114, 135)
                colors['yellow'] = (110, 193, 200)
                colors['white'] = (173, 193, 199)
                colors['pink'] = (169, 161, 227)
                colors['grey'] = (185, 184, 183)
                # means=[]

                # for ell in ellipses:
                #     cv2.ellipse(frame, ell, (0, 250, 0), 1)

                for crc in shapes:
                    mask = np.zeros((480, 640), np.uint8)
                    (x, y), radius = cv2.minEnclosingCircle(crc)
                    center = (int(x), int(y))
                    radius = int(radius)
                    centerTxt = (int(x - radius), int(y + radius + 10))

                    ellipse = cv2.fitEllipse(crc)
                    cv2.ellipse(frame, ellipse, (0, 250, 0), 1)

                    #cv2.circle(frame, center, radius + 3, (0, 250, 0), 1)
                    cv2.drawContours(mask, [crc], 0, 255, -1)
                    mean = cv2.mean(frame, mask=mask)
                    means = (int(mean[0]), int(mean[1]), int(mean[2]))
                    i = 15
                    for keys, m in colors.iteritems():
                        if means[0] <= m[0] + i and means[0] >= m[0] - i and means[1] <= m[1] + \
                                i and means[1] >= m[1] - i and means[2] <= m[2] + i and means[2] >= m[2] - i:
                            print("mean:", means)
                            # print "index:", keys
                            # means.append(mean)

                    cv2.drawContours(frame, [crc], 0, (0, 250, 0), cnt_style)
                    #print("xy:", x, y)

                    cv2.imwrite('img//ok//frame' + str(count) + '.jpg', frame)
                    cv2.imshow('ok', frame)
                    count += 1

                    means = []
                    means.append(str("%3d" % mean[0]))
                    means.append(str("%3d" % mean[1]))
                    means.append(str("%3d" % mean[2]))

                    cv2.putText(frame, str(means), centerTxt,
                                font, .4, (0, 250, 0), 1)
                    # print "color:", mean
                # print "means:", means

                for cnt in defects:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    centerTxt = (int(x + radius - 10), int(y - radius + 10))
                    radius = int(radius)
                    #cv2.circle(frame, center, radius + 3, (0, 0, 250), 1)
                    if len(cnt) > 5:
                        ellipse = cv2.fitEllipse(cnt)
                        cv2.ellipse(frame, ellipse, (0, 0, 250), 1)
                        # if y > 200 and y < 350 and x > 200 and x < 350:
                        cv2.imwrite(
                            'img//erro//frame' + str(count) + '.jpg', frame)
                        cv2.imshow('erro', frame)
                        count += 1
                    txt = str(int(x)) + "x" + str(int(y))
                    #cv2.putText(frame, txt, centerTxt, font, .3, (255, 255, 255), 1)
                # draw shape with cotamination
                for ctm in contaminations:
                    (x, y), radius = cv2.minEnclosingCircle(ctm)
                    centerA = (int(x), int(y))
                    for crc in shapes:
                        (x, y), radius = cv2.minEnclosingCircle(crc)
                        centerB = (int(x), int(y))
                        radius = int(radius)
                        dist = cv2.norm(centerA, centerB)
                        if dist < radius:
                            cv2.circle(
                                frame, centerB, radius + 3, (0, 0, 250), 1)
                            txt = str(int(x)) + "x" + str(int(y))
                            centerTxt = (int(x + radius - 10),
                                         int(y - radius + 10))
                            cv2.putText(
                                frame, txt, centerTxt, font, .3, (255, 255, 255), 1)

                cv2.drawContours(
                    frame, contaminations, -1, (0, 0, 250), cnt_style)
                last_inspect = cv2.getTickCount()

        # time fps
        e2 = cv2.getTickCount()
        t = (e2 - e1) / cv2.getTickFrequency()
        fps = str("%.2d" % (1 / t))
        msg = "fps: " + str(fps)
        t = str("%.3f" % t)
        msg = "time (s): " + str(t)
        cv2.rectangle(frame, (0, 0), (90, 14), (0, 0, 0), -1)

        cv2.putText(frame, t + 's ' + fps + 'fps', (1, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 250, 0), 1)
        text = "ok:[" + \
            str(total_shapes) + "] error:[" + str(total_defects) + "]"
        cv2.putText(frame, text, (90, 10), font, .4, (0, 250, 0), 1)

        cv2.imshow('frame', frame)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
            cv2.destroyAllWindows()
