catkin_make를 할 때, 맨 처음 catkin_make는 "catkin_make -DPYTHON_EXECUTABLE=<파이썬 가상환경 경로>"로 실행되는 가상환경을 지정할 것.

venv를 실행할 때의 파이썬 가상환경 경로는 import sys; print(sys.executable) 등의 파이썬 스크립트로 출력할 수 있음.

현재 이것을 작성하고 있는 로컬 컴퓨터 기준으로는 /home/tepk2924/tepk2924Works/myGraduationProject/venv/bin/python가 출력되므로, 맨 처음 catkin_make의 명령어는 다음과 같음: 

"catkin_make -DPYTHON_EXECUTABLE=/home/tepk2924/tepk2924Works/myGraduationProject/venv/bin/python"

```bash
catkin_make -DCATKIN_WHITELIST_PACKAGES="irb120_description;irb120_moveit_config;arm_pkg;robotiq_epick_gripper_control" -DPYTHON_EXECUTABLE=/home/riseabb/johan_ws/myGraduationProject/venv/bin/python3
```

## Robotiq Usage

* First, run roscore or roslaunch
```bash
roscore
```

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