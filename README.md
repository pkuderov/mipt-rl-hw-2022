# Repository for CDS MIPT RL 2022 course assignments
2. DQN


Команда для логирования:
tensorboard --logdir="../../data" --host localhost --port 8088


2.1
График Обучения для алгоритма DQN в среде Lunar.
Измененный параметр 'learning_starts': 10000
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/Lunar_DQN_1.png)
2.2
График DDQN. Измененный параметр 'learning_starts': 10000. Здесь можно заметить, что обновление весов 
у target net лишь в конце эпизода помогает избегать резких скачков в графике и помогает сделать обучение более гладким.
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/Lunar_DDQN_1.png)
2.3
График обучения DQN. Измененный параметр 'learning_starts': 10000. 'batch_size': 64.
Здесь заметно, что увеличение размера батча способствовало более быстрой оптимизации функции вознаграждения(за меньшее число шагов)
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/Lunar_DQN_2.png)


3. Актор критик
Из ниже представленных графиков реализации обучения, можно сделать вывод, что слишком быстрое чередование обновлений критика и актора
способствует плохому обучению. Это происходит потому что как для одного так и для другого не достаточно малого числа шагов, чтобы получать
либо примлимую оценку Функции полезности либо приемлимое распределение Действий. Но как только мы даем большее число шагов на обучение критика(второй пример) то после того как он начинает давать хорошую оценку функции полезности, актор на этой оценке начинает хорошо обучаться.

python hw3/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/AC_eval_1.png)
python hw3/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 100
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/AC_eval_2.png)
python hw3/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 10 -ngsptu 10
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/AC_eval_3.png)
