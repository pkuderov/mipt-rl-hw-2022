# Repository for CDS MIPT RL 2022 course assignments
2.3 
Графики средней отдачи при маленьком батче(train_batch = 100, eval_batch = 20)



Графики средней отдачи при большом батче(train_batch = 1000, eval_batch = 400)



Команда для запуска обучения:
python run_hw2.py --env_name CartPole-v0
Команда для логирования:
tensorboard --logdir="../../data" --host localhost --port 8088