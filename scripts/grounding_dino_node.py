#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from groundingdino.util.inference import load_model, predict
import torch

class GroundingDINONode:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                "groundingdino_swint_ogc.pth")
        rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.results_pub = rospy.Publisher("/groundingdino/results", String, queue_size=10)
        rospy.loginfo("GroundingDINO ROS Node initialized.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            prompt = "a dog, a cat, a person"  # or make it a ROS param
            boxes, logits, phrases = predict(self.model, cv_image, prompt, box_threshold=0.3, text_threshold=0.25)
            result_str = ', '.join(phrases)
            self.results_pub.publish(result_str)
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

if __name__ == '__main__':
    rospy.init_node('grounding_dino_node', anonymous=True)
    node = GroundingDINONode()
    rospy.spin()
