## 2. Практическая часть
#### Эксперимент 1: CartPole
Кривые обучения средней отдачи на каждой итерации для экспериментов с маленьким размером пакета:
![Experiment 1 small](pictures/experiment_1_small.png)
![Image 1 legend](pictures/legend_1.png)

Кривые обучения средней отдачи на каждой итерации для экспериментов с большим размером пакета:
![Experiment 1 large](pictures/experiment_1_large.png)
![Image 2 legend](pictures/legend_2.png)

Ответы на вопросы:
1) Какой из вариантов оценки отдачи имеет лучшие результаты без нормализации значения преимущества?
- Лучшим результатом является q1_lb_rtg_dsa (-b 5000 -rtg -dsa)

2) Помогает ли нормализация значения преимущества?
- При маленьком размере пакета нормализация не даёт ощутимого выигрыша, а при большом – видно, что решение получается с меньшей дисперсией

3) Как влияет размер пакета на качество обучения?
- Чем больше размер пакета, тем больше скорость сходимости решения
```
!python /content/cds_rl_2022/rl_hw/hw2/hw2/scripts/run_hw2.py \
    --env_name CartPole-v0 -n 100 -b 1000 \
    -dsa --exp_name q1_sb_no_rtg_dsa

!python /content/cds_rl_2022/rl_hw/hw2/hw2/scripts/run_hw2.py \
    --env_name CartPole-v0 -n 100 -b 1000 \
    -rtg -dsa --exp_name q1_sb_rtg_dsa

!python /content/cds_rl_2022/rl_hw/hw2/hw2/scripts/run_hw2.py \
    --env_name CartPole-v0 -n 100 -b 1000 \
    -rtg --exp_name q1_sb_rtg_na

!python /content/cds_rl_2022/rl_hw/hw2/hw2/scripts/run_hw2.py \
    --env_name CartPole-v0 -n 100 -b 5000 \
    -dsa --exp_name q1_lb_no_rtg_dsa

!python /content/cds_rl_2022/rl_hw/hw2/hw2/scripts/run_hw2.py \
    --env_name CartPole-v0 -n 100 -b 5000 \
    -rtg -dsa --exp_name q1_lb_rtg_dsa

!python /content/cds_rl_2022/rl_hw/hw2/hw2/scripts/run_hw2.py \
    --env_name CartPole-v0 -n 100 -b 5000 \
    -rtg --exp_name q1_lb_rtg_na
```
#### Эксперимент 2: InvertedPendulum
Подбираем параметры таким образом, чтобы размер пакета был минимальным, а скорость обучения была бы максимальной:
![Experiment 2](pictures/experiment_2_search.png)
![Image 3 legend](pictures/legend_3.png)

Лучший результат получился при b = 1000, lr = 0.08:
```
!python /content/cds_rl_2022/rl_hw/hw2/hw2/scripts/run_hw2.py \
    --env_name InvertedPendulum-v2 \
    --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 \
    -b 1000 -lr 0.08 -rtg \
    --exp_name q2_b1000_r0.08
```
![Best of experiments 2](pictures/experiment_2_final.png)

#### Эксперимент 3: LunarLander
![Experiment 3](pictures/experiment_3.png)

#### Эксперимент 4: HalfCheetah
Подбор наилучших параметров:
![Experiment 4 search](pictures/experiment_4_search.png)
![Image 6 legend](pictures/legend_6.png)
Чем больше learning_rate, тем больше награда, получаемая в итоге

Серия экспериментов с парамтерами b=30000, lr=0.02:
![Experiment 4](pictures/experiment_4_final.png)
![Image 7 legend](pictures/legend_7.png)

#### Эксперимент 5: Hopper
Обучение агента с разными параметрами λ ∈ [0, 0.95, 0.99, 1]:
![Experiment 5](pictures/experiment_5.png)
![Image 8 legend](pictures/legend_8.png)
Чем больше λ, тем больше награда, получаемая в итоге