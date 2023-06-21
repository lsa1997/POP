## Pretrained POP Models
We have released pretrained POP models [here](https://drive.google.com/drive/folders/1x_ujxvyOlPyfopdFjikuMx4bpilNbAiG?usp=sharing). It includes: 
- models trained on base classes,
- models updated on novel classes under 1-/5-/10-shot settings with different random seeds.

Note that all models are trained based on the released code. We fix an improper setting in the cropping operation and achieve results better than those reported in our paper.

## Performance on PASCAL-5i
For each few-shot setting, we release three models trained with different random seeds, i.e., different support images. They are all finetuned based on the released base models.
|     1-shot     |  Base |  Novel  |     Total    |
| :-------------: | :---------: | :--------: | :--------: |
| Seed-123 |    73.86     |    33.90   |     64.35    |
| Seed-234 |    73.77     |    37.71   |     65.18    |
| Seed-345 |    73.79     |    38.61   |     65.41    |
| Average  |    73.81     |    36.74   |     64.98    |

|     5-shot     |  Base |  Novel  |     Total    |
| :-------------: | :---------: | :--------: | :--------: |
| Seed-123 |    74.57     |    57.10   |     70.41    |
| Seed-234 |    74.82     |    57.21   |     70.63    |
| Seed-345 |    74.83     |    55.34   |     70.19    |
| Average  |    74.74     |    56.55   |     70.41    |

|     10-shot     |  Base |  Novel  |     Total    |
| :-------------: | :---------: | :--------: | :--------: |
| Seed-123 |    75.20     |    60.51   |     71.70    |
| Seed-234 |    75.11     |    58.79   |     71.23    |
| Seed-345 |    75.14     |    60.87   |     71.74    |
| Average  |    75.15     |    60.06   |     71.56    |

## Performance on COCO-20i
The released models on COCO are finetuned on few-shot data with batch size 1 and fp16 training.

|     1-shot     |  Base |  Novel  |     Total    |
| :-------------: | :---------: | :--------: | :--------: |
| Seed-123 |    54.31     |    24.61   |     46.97    |
| Seed-234 |    54.18     |    23.16   |     46.52    |
| Average  |    54.24     |    23.89   |     46.75    |

|     5-shot     |  Base |  Novel  |     Total    |
| :-------------: | :---------: | :--------: | :--------: |
| Seed-123 |    54.59     |    36.81   |     50.20    |
| Seed-234 |    54.58     |    36.72   |     50.17    |
| Average  |    54.59     |    36.77   |     50.19    |

|     10-shot     |  Base |  Novel  |     Total    |
| :-------------: | :---------: | :--------: | :--------: |
| Seed-123 |    54.66     |    39.06   |     50.81    |
| Seed-234 |    54.57     |    38.76   |     50.67    |
| Average  |    54.62     |    38.91   |     50.74    |