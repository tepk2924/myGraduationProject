#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
# import tensorflow as tf

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(20) # 10hz
    i = 0
    while not rospy.is_shutdown():
        i += 0.1
        # ten = tf.constant(i)
        # strin = f"{ten}"
        strin = input("Input : ")
        rospy.loginfo(strin)
        pub.publish(strin)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass