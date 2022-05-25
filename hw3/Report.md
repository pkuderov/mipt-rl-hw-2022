# Домашняя работа  Практическая часть 2
### Мелкумов Михаил

## (2.1) Тестирование Q-обучения
<img src="3h.jpg" height="150">

###  Качество базовой версии Q-обучения (Ms. Pac-Man) 


<img src="Pacman1.png" height="350">
<img src="Pacman2.png" height="350">

###  Качество двойного Q-обучения (LunarLander)

3 исследования DQN:


<img src="dqn_moon.png" height="350">


3 исследования DDQN:


<img src="dqn_moon2.png" height="350">

Сравнение лучших dqn(оранжевый) и ddqn(серый):


<img src="dqn_moon3.png" height="350">

###  Эксперименты с гиперпараметрами

Я решил попробовать поменять количество слоёв в сети

<img src="dqn_param.png" height="350">

И не нашел особой разницы в результатах(

## (2.2) Тестирование метода актор-критик

### Проверка правильности реализации на CartPole

<img src="ac_param.png" height="350">

Наилучший результат показали параметры:
```
num_target_updates: 10
num_grad_steps_per_target_update: 10
```

Протестируем на других средах с такими же параметрами

### Проверка метода актор-критик на более сложных средах

HalfCheet:

<img src="hach2.png" height="350">

InvertedPendulum:

<img src="pend.png" height="350">