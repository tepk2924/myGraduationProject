import pyzed.sl as sl
import numpy as np
import cv2
import rospy
import inspect
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

def zed_camera():
    # Create a Camera object
    zed = sl.Camera()
    pub_rgb = rospy.Publisher('zed_rgb_frame', Image, queue_size=10)
    pub_depth = rospy.Publisher('zed_depth_frame', Float32MultiArray, queue_size=10)
    rospy.init_node('camera')
    bridge = CvBridge()
    rate = rospy.Rate(10)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY  # Use QUALITY depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.enable_fill_mode = True
    
    depth = sl.Mat()
    rgb_left = sl.Mat()
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))

    info = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    for name, value in inspect.getmembers(info):
        if not name.startswith("_") and not inspect.ismethod(value):
            print(f'{name}: {value}')
    print(f'height: {info.image_size.height}')
    print(f'width: {info.image_size.width}')

    while not rospy.is_shutdown():
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_image(rgb_left, sl.VIEW.LEFT)

            # depth_np = depth.get_data()
            rgb_left_np = rgb_left.get_data()[:, :, :3]
            depth_np:np.ndarray = depth.get_data()
            depth_msg = Float32MultiArray()
            depth_msg.data = depth_np.reshape((-1)).tolist()
            pub_rgb.publish(bridge.cv2_to_imgmsg(rgb_left_np, "rgb8"))
            pub_depth.publish(depth_msg)
            rate.sleep()

    # Close the camera
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    zed_camera()