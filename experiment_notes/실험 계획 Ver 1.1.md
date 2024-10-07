# 졸업논문연구 실험 계획
#### Unet 및 Suction-Grasp Net을 이용한 Suction Cup 파지점을 Scene에서 찾기 및 파지 실험

+ 실험 제안
    + Scene : 물체가 5개 놓여있음, Valid한 물체가 적어도 하나 섞임
    + 실험 방법 : 각 Scene마다 파지 진행, 로봇이 각 Scene에서 3번 연속으로 더 이상 유효한 파지점이 없다고 판단하거나, 3번 연속으로 파지 실패 시 다음 Scene으로 넘어감
    + 각 Scene마다 맨 처음에 어떤 물건이 올려져 있는지, 각 파지 시도마다 어떤 object를 파지 시도하였는지, 그 시도가 성공적이었는지 등을 모두 기록.
    + 성공률 추산 :
    $$V_i: \rm 각 \ Scene에서의 \ Valid \ Object의 \ 개수$$
    $$S_i: \rm 각 \ Scene에서 \ 성공적으로 \ 파지한 \ Valid \ Object의 \ 개수$$
    $$N: \rm 전체 \ Scene의 \ 수$$
    $$S_{initial}: \rm 각 \ Scene에서 \ 첫 \ 파지의 \ 총 \ 성공 \ 횟수$$
    $$SR_{initial} = \frac{S_{initial}}{N}$$
    $$SR_{avg} = \frac{\sum_{i=1}^N \frac{S_i}{V_i}}{N}$$
    $$SR_{total}=\frac{\sum_{i=1}^N S_i}{\sum_{i=1}^N V_i}$$