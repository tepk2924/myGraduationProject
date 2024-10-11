import os

with open(os.path.join(os.path.dirname(__file__), "resultwithuneta4e29.txt"), "r") as f:
    lines = f.readlines()

infos = []

for line in lines:
    infos.append(line.strip().split('|'))
    infos[-1][1] = int(infos[-1][1])
    infos[-1][3] = int(infos[-1][3])
    infos[-1][4] = int(infos[-1][4])
    infos[-1][5] = int(infos[-1][5])
    infos[-1][6] = int(infos[-1][6])

rate_cum = 0
V_cum = 0
I_cum = 0
S_cum = 0
F_cum = 0

for info in infos:
    rate_cum += info[5]/info[3]
    V_cum += info[3]
    I_cum += info[4]
    S_cum += info[5]
    F_cum += info[6]

print(f"SR_avggrasp = {rate_cum/20}")
print(f"SR_totalgrasp = {S_cum/V_cum}")
print(f"SR_totalavoid = {1 - F_cum/I_cum}")