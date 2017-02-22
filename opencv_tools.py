# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:14:11 2017

@author: amal
"""

import cv2

class frame_feed(object):#object to fetch frame from webcam
    
    def __init__(self,cascadeFile=None,fullscreen=True,device=0,mirror=True,header='Camera Output'):
        self.cap = cv2.VideoCapture(device)
        self.frame=None
        self.mirror=mirror
        self.header=header
        if cascadeFile!=None:
            self.cascade = cv2.CascadeClassifier(cascadeFile)
        if fullscreen==True:
            cv2.namedWindow(header, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(header, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        self.fetch()    

    def fetch(self):#read and fetch new frame
        ret, self.frame = self.cap.read()
        if self.mirror==True:
             self.frame=cv2.flip(self.frame,1)
        return self.frame
    
    def current_frame(self):
        return self.frame
    
    def draw_rects(self, rects, color=(0,255,0)):#function for drawing rectangles(rects=x1,y1,yx2,y2)
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)   
    
    def detect(self, minimumFeatureSize=(80,80)):#function for calling haar face detector

        img=cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cascade=self.cascade        
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2] 
        return rects
    def display(self):    
        cv2.imshow(self.header,self.frame) 
        self.fetch()
    def halt(self):
        self.cap.release()
        for i in xrange(10):  #exit  code
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        
        