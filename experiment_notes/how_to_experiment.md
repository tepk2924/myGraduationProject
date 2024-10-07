## Launch ROS

* Change directory
```bash
cd ros_ws/src/arm_pkg/launch
```

* Launch rviz, nodes, camera, robot arm
```bash
roslaunch arm_real.launch
```

## Robotiq (Suction Cup) Usage

* Launch robotiq epick driver node
```bash
rosrun robotiq_epick_gripper_control robotiq_epick_node.py /dev/ttyUSB0
```
* Run simple CLI
```bash
rosrun robotiq_epick_gripper_control robotiq_epick_simplecontroller_node.py
```

* Add permissions

```bash
sudo usermod -a -G dialout $USER
```

```bash
sudo chmod 777 /dev/ttyUSB0
```

## Execute

* Suction Grasp Execute
```bash
rosrun arm_pkg execute.py
```