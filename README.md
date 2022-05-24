# Repository for CDS MIPT RL 2022 course assignments
##HW2:

Команда для логирования:

tensorboard --logdir="../../data" --host localhost --port 8088

2.3 
Cart-Pole экспермент:
Команда для запуска обучения:

python run_hw2.py --env_name CartPole-v0

Графики средней отдачи при маленьком батче(train_batch = 100, eval_batch = 20)
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/2.3_small_batch_val.png)
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/2.3_small_batch_train.png)


Графики средней отдачи при большом батче(train_batch = 1000, eval_batch = 400)
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/2.3_big_batch_val.png)
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/2.3_big_batch_train.png)


InvertedPendulum эксперемент:

Команда для запуска обучения:
python run_hw2.py --env_name InvertedPendulum-v2

Средняя отдача ушла в 1000 при значении train_batch = 1000, eval_batch = 400
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/2.3_invert_eval.png)
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/2.3_invert_train.png)


baseline LunaLander: 

Команда для запуска обучения:
python run_hw2.py --env_name LunarLanderContinuous-v2 --nn_baseline --reward_to_go

средняя отдача около 50 (train_batch = 1000, eval_batch = 400)
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/2.3_luna_batch_eval.png)
![Image alt](https://github.com/Gricha1/mipt-rl-hw-2022/raw/HW_2_Gorbov/images/2.3_luna_batch_train.png)

Gae выполнен в виде кода.
