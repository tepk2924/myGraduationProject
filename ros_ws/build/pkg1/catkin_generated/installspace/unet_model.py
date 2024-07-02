#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import tensorflow as tf
import importlib
import sys
import os
import yaml
import glob
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy import signal

class unet_model:
    def __init__(self):
        self.pub = rospy.Publisher("result", Image, queue_size=10)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("camera_frame", Image, self.callback)
        model_name = rospy.get_param("model_name")
        target_epoch = rospy.get_param("target_epoch")
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        sys.path.append(project_dir)
        arch = importlib.import_module(f"unet_checkpoints.{model_name}.unet_architecture")
        saved_model_dir = os.path.join(os.path.join(project_dir, "unet_checkpoints"), model_name)
        with open(os.path.join(saved_model_dir, "hyperparameters.yml"), 'r') as f:
            hyperparameters = yaml.safe_load(f)
        inputs, outputs = arch.build_unet_graph(hyperparameters)
        self.model = arch.Unet(inputs, outputs)
        weight_filelist = glob.glob(os.path.join(saved_model_dir, "weights/*.h5"))
        weight_filelist.sort()
        epoch_list = [int(os.path.basename(weight_file).split('-')[0]) for weight_file in weight_filelist]
        epoch_list = np.array(epoch_list)
        if target_epoch == -1:
            weight_file = weight_filelist[-1]
            target_epoch = epoch_list[-1]
        else:
            idx = np.where(epoch_list == target_epoch)[0][0]
            weight_file = weight_filelist[idx]
            target_epoch = epoch_list[idx]
        self.model.load_weights(weight_file)

    def callback(self, data):
        cv_image:np.ndarray = self.bridge.imgmsg_to_cv2(data, "bgr8")

        image_height = cv_image.shape[0]
        image_width = cv_image.shape[1]

        laplacian = np.array([[1, 4, 1],
                              [4,-20, 4],
                              [1, 4, 1]])
        RGBE_normalized = np.empty((image_height, image_width, 4), dtype=np.float)
        RGBE_normalized[:, :, :3] = cv_image/255.

        kernaled = (signal.convolve2d(cv_image[:, :, 0], laplacian, mode="same", boundary="symm")**2 + 
                    signal.convolve2d(cv_image[:, :, 1], laplacian, mode="same", boundary="symm")**2 + 
                    signal.convolve2d(cv_image[:, :, 2], laplacian, mode="same", boundary="symm")**2)
        RGBE_normalized[:, :, 3] = (kernaled - np.mean(kernaled))/np.std(kernaled)

        RGBE_normalized = tf.expand_dims(tf.convert_to_tensor(RGBE_normalized, dtype=tf.float32), axis=0)

        logit = self.model(RGBE_normalized, training=False)
        pred_segmap_prob = tf.nn.softmax(logit)
        pred = tf.argmax(pred_segmap_prob, axis=-1)
        pred_segmap=tf.squeeze(pred, axis=0).numpy()
    
        convert_RGB = np.array([[255, 0, 0],
                                [0, 255, 0],
                                [0, 0, 255]], dtype=np.uint8)
        
        pred_segmap_RGB = convert_RGB[pred_segmap]
        self.pub.publish(self.bridge.cv2_to_imgmsg(pred_segmap_RGB, "bgr8"))

def main():
    model = unet_model()
    rospy.init_node("unel_model", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down by the KeyBoard")
    except CvBridgeError:
        print("Shutting Down by Cv_BridgeError")
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()