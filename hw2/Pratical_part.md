# Отчёт по выполнению практической части

## Эксперимент 1: CartPole
1. Small batch size
   
   ![exp1_1_plot](./images/exp1_1_plot.jpg)
   ![exp1_1_legend](./images/exp1_1_legend.jpg)
2. Large batch size
   
   ![exp1_2_plot](./images/exp1_2_plot.jpg)
   ![exp1_2_legend](./images/exp1_2_legend.jpg)

- Какой из вариантов оценки отдачи имеет лучшие результаты без нормализации
значения преимущества?

**Ответ**: Наилучший результат при параметрах: ```reward_to_go = True```, ```standardize_advantages = False``` и batch_size = 5000 (малиновая линия на втором графике).

- Помогает ли нормализация значения преимущества?

**Ответ**: Да, при большом размере пакета.

- Как влияет размер пакета на качество обучения?

**Ответ**: Увеличение размера пакета положительно влияет на скорость обучения.

## Эксперимент 2: InvertedPendulum
Наилучшие полученные значения: batch_size = 1000, learning rate = 0.02

![exp2_plot](./images/exp2_plot.jpg)

Команда запуска:

```
!python scripts/run_hw2.py \
    --env_name InvertedPendulum-v2 \
    --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 \
    -b 1000 -lr 0.02 -rtg \
    --exp_name q2_b1000_r0.02
```

## Эксперимент 3: LunarLander

![exp3](./images/exp3.jpg)

## Эксперимент 4: HalfCheetah

![exp4_1_plot](./images/exp4_1_plot.jpg)
![exp4_1_legend](./images/exp4_1_legend.jpg)

(* допустила опечатку в названиях логов: последние три эксперимента относятся к batch size = 50000)

Увеличение коэффициента скорости обучения значительно ускоряет сходимость обучения, увеличение размера также увеличивает награду.

Наилучшие полученные значения: batch_size = 30000, learning rate = 0.02

![exp4_2_plot](./images/exp4_2_plot.jpg)
![exp4_2_legend](./images/exp4_2_legend.jpg)

## Эксперимент 5: Hopper

![exp5_plot](./images/exp5.jpg)
![exp5_legend](./images/exp5_legend.jpg)

λ близкий к единице ускоряет сходимость к решению. 
