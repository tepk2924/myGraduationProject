# 졸업논문연구 실험 계획
#### Unet 및 Suction-Grasp Net을 이용한 Suction Cup 파지점을 Scene에서 찾기 및 파지 실험

+ 실험 제안
    + Scene : 물체가 5개 놓여있음, Valid한 물체가 적어도 하나 섞임
    + 실험 방법 : 각 Scene마다 파지 진행, 로봇이 각 Scene에서 3번 연속으로 더 이상 유효한 파지점이 없다고 판단하거나 파지 실패 시 다음 Scene으로 넘어감
    + 각 Scene마다 맨 처음에 어떤 물건이 올려져 있는지, 각 파지 시도마다 어떤 object를 파지 시도하였는지, 그 시도가 성공적이었는지 등을 모두 기록.
    + 성공률 추산 :
    $$V_i: \rm 각 \ Scene에서의 \ Valid \ Object의 \ 개수$$
    $$I_i: \rm 각 \ Scene에서의 \ Invalid \ Object의 \ 개수 \ (=5 \it - V_i)$$
    $$S_i: \rm 각 \ Scene에서 \ 성공적으로 \ 파지한 \ Valid \ Object의 \ 개수$$
    $$F_i: \rm 각 \ Scene에서 \ 파지 \ 시도한 \ Invalid \ Object의 \ 개수$$
    $$N: \rm 전체 \ Scene의 \ 수$$
    $$S_{initial}: \rm 각 \ Scene에서 \ 첫 \ 파지 \ 시도의 \ 총 \ 성공 \ 횟수$$
    $$SR_{initial} = \frac{S_{initial}}{N}$$
    $$SR_{avggrasp} = \frac{\sum_{i=1}^N (S_i/V_i)}{N}$$
    $$SR_{totalgrasp}=\frac{\sum_{i=1}^N S_i}{\sum_{i=1}^N V_i}$$
    $$SR_{totalavoid}=1 - \frac{\sum_{i=1}^N F_i}{\sum_{i=1}^N I_i}$$

+ 실험 결과
    1. Without Unet
        - Each Scene
            |Scene idx|First Attempt|$V_i$|$I_i$|$S_i$|$F_i$|
            |:---:|:---:|:---:|:---:|:---:|:---:|
            |1|Y|5|0|2|0|
            |2|N|1|4|0|2|
            |3|N|3|2|2|2|
            |4|Y|3|2|3|2|
            |5|N|1|4|0|2|
            |6|Y|4|1|1|1|
            |7|Y|5|0|4|0|
            |8|Y|5|0|3|0|
            |9|N|1|4|1|1|
            |10|N|2|3|0|2|
            |11|Y|2|3|1|3|
            |12|N|2|3|0|2|
            |13|N|5|0|3|0|
            |14|N|4|1|1|0|
            |15|N|2|3|0|2|
            |16|Y|5|0|2|0|
            |17|N|4|1|1|1|
            |18|Y|4|1|3|1|
            |19|Y|5|0|3|0|
            |20|N|3|2|0|1|
        - Performance
            |$SR_{initial}$|$SR_{avggrasp}$|$SR_{totalgrasp}$|$SR_{totalavoid}$|
            |:---:|:---:|:---:|:---:|
            |0.45|0.4033|0.4545|0.3529|

    2. Unet A4E29 (Conservative)
        - Each Scene
            |Scene idx|First Attempt|$V_i$|$I_i$|$S_i$|$F_i$|
            |:---:|:---:|:---:|:---:|:---:|:---:|
            |1|Y|5|0|1|0|
            |2|Y|1|4|1|0|
            |3|Y|3|2|1|1|
            |4|Y|3|2|2|0|
            |5|N|1|4|0|1|
            |6|Y|4|1|1|0|
            |7|Y|5|0|2|0|
            |8|Y|5|0|1|0|
            |9|Y|1|4|1|0|
            |10|N|2|3|0|0|
            |11|Y|2|3|1|1|
            |12|N|2|3|1|1|
            |13|Y|5|0|2|0|
            |14|Y|4|1|2|0|
            |15|Y|2|3|2|0|
            |16|Y|5|0|4|0|
            |17|Y|4|1|2|0|
            |18|Y|4|1|3|0|
            |19|Y|5|0|3|0|
            |20|Y|3|2|2|0|
        - Performance
            |$SR_{initial}$|$SR_{avggrasp}$|$SR_{totalgrasp}$|$SR_{totalavoid}$|
            |:---:|:---:|:---:|:---:|
            |0.85|0.5133|0.4848|0.8823|