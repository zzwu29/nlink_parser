import cv2
import numpy as np
import rospy
from numpy import random
from nlink_parser.msg import TofsenseMFrame0 


class TofsenseMFrame0Viz:
    def __init__(self, pixel_num = 8):
        rospy.init_node('TofsenseMFrame0Viz')

        self.max_distance = 6.0

        self.height, self.width = pixel_num, pixel_num
        self.image_range = 255*np.ones((self.height, self.width, 3), dtype=np.uint8)
        self.image_status = 255*np.ones((self.height, self.width, 3), dtype=np.uint8)
        self.image_rssi = 255*np.ones((self.height, self.width, 3), dtype=np.uint8)
        
        self.image_range_viz = cv2.resize(self.image_range, (self.height*100, self.width*100))
        self.image_status_viz = cv2.resize(self.image_status, (self.height*100, self.width*100))
        self.image_rssi_viz = cv2.resize(self.image_rssi, (self.height*100, self.width*100))
        self.images_viz = np.hstack((self.image_range_viz, self.image_status_viz, self.image_rssi_viz))

        cv2.namedWindow("Tofsense M Frame Ranging",0)
        cv2.resizeWindow("Tofsense M Frame Ranging", 1500, 500)
        cv2.imshow('Tofsense M Frame Ranging', self.images_viz)
        cv2.waitKey(10)

        self.tofsensem_frame0_sub = rospy.Subscriber('/nlink_tofsensem_frame0', TofsenseMFrame0, self.tofsensem_frame0_cbk, queue_size=100000) 
    
    def tofsensem_frame0_cbk(self, ranging_msg):
        # print(ranging_msg.id, ranging_msg.system_time, ranging_msg.pixel_count, len(ranging_msg.pixels))
        self.image_status = 255*np.ones((self.height, self.width, 3), dtype=np.uint8)
        self.image_rssi = 255*np.ones((self.height, self.width, 3), dtype=np.uint8)

        for i in range(self.height * self.width):
            if i >= len(ranging_msg.pixels):
                break
            dis = np.round(0.001 * ranging_msg.pixels[i].dis, 2)
            B, G ,R = self.jet_mapping(np.round(dis/self.max_distance*255))
            self.image_range[i//self.width, i%self.width] = np.array([B,G,R])
            status = ranging_msg.pixels[i].dis_status
            # opencv mat (height, width, channel)
            if status == 0:
                self.image_status[i//self.width, i%self.width] = np.array([0, 255, 0])
            else:
                self.image_status[i//self.width, i%self.width] = np.array([0, 0, 255])


        self.image_range_viz = cv2.resize(self.image_range, (self.height*100, self.width*100))
        self.image_status_viz = cv2.resize(self.image_status, (self.height*100, self.width*100))
        self.image_rssi_viz = cv2.resize(self.image_rssi, (self.height*100, self.width*100))
        cv2.putText(self.image_range_viz, "t: " + str(ranging_msg.system_time), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2, cv2.LINE_AA) # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
        for i in range(self.height * self.width):
            if i >= len(ranging_msg.pixels):
                break
            dis = np.round(0.001 * ranging_msg.pixels[i].dis, 2)
            # so here is (v, u) not (u, v)
            cv2.putText(self.image_range_viz, str(dis), (int((i%self.width+0.5)*100), int((i//self.width+0.5)*100)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA) # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
            cv2.putText(self.image_status_viz, str(ranging_msg.pixels[i].dis_status), (int((i%self.width+0.5)*100), int((i//self.width+0.5)*100)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA) # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
            cv2.putText(self.image_rssi_viz, str(ranging_msg.pixels[i].signal_strength), (int((i%self.width+0.5)*100), int((i//self.width+0.5)*100)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA) # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576

        self.images_viz = np.hstack([self.image_range_viz, self.image_status_viz, self.image_rssi_viz])
        cv2.imshow('Tofsense M Frame Ranging', self.images_viz)
        cv2.waitKey(10)
    
    def jet_mapping(self, intensity):
        # https://blog.csdn.net/qq_41498261/article/details/109603986
        B, G, R = 0, 0, 0
        if intensity >= 0 and intensity < 32:
            B = 128 + 4*intensity
        elif intensity < 96:
            B = 255
            G = 4*(intensity - 32)
        elif intensity < 160:
            B = 254 - 4*(intensity - 96)
            G = 255
            R = 2 + 4*(intensity - 96)
        elif intensity < 224:
            B = 0
            G = 252 - 4*(intensity - 160)
            R = 255
        elif intensity < 256:
            B = 0
            G = 0
            R = 252 - 4*(intensity - 224)
        
        return B,G,R
        
    def spin(self):
        rospy.spin()



if __name__ == '__main__':
    tofsense_mframe0_viz = TofsenseMFrame0Viz()
    tofsense_mframe0_viz.spin()

