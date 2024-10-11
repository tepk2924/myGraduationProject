import os
import random

direc = os.path.dirname(__file__)

N = 20

with open(os.path.join(direc, "valid_objs.txt"), "r") as f:
    valid_list = [ele.rstrip() for ele in f.readlines()]

with open(os.path.join(direc, "invalid_objs.txt"), "r") as f:
    invalid_list = [ele.rstrip() for ele in f.readlines()]

with open(os.path.join(direc, "experiment_list_tmp.md"), "w") as f:
    for idx in range(1, N + 1):
        valid_num = random.randint(1, 5)
        valid_objs = random.sample(valid_list, valid_num)
        invalid_objs = random.sample(invalid_list, 5 - valid_num)
        f.write(f"* [ ] scene #{idx:03d}: {valid_num} valid, {5 - valid_num} invalid\n")
        f.write(f"  + [ ] first attempt grasp\n")
        f.write(f"  + valid objs:\n")
        for obj in valid_objs:
            f.write(f"    - [ ] successfully grasped {obj}\n")
        if valid_num != 5:
            f.write(f"  + invalid objs:\n")
            for obj in invalid_objs:
                f.write(f"    - [ ] tried to grasp invalid {obj}\n")