# Paper model vs our model

Same data (`data/`), same superpixels (np=10), same splits, same hyper-parameters. Mean ± std over seeds.

## 1. Main comparison

| Model | Split | AUC | OA | PRECISION | RECALL | F1 | MCC | KAPPA |
|---|---|---|---|---|---|---|---|---|
| *Published in the paper (different data)* | random 70/30 | 0.9838 | 0.9838 | 0.9772 | 0.9907 | 0.9839 | 0.9676 | 0.9675 |
| Paper model | random | 0.9998 ± 0.0001 | 0.9949 ± 0.0011 | 0.9952 ± 0.0015 | 0.9961 ± 0.0029 | 0.9957 ± 0.0010 | 0.9894 ± 0.0023 | 0.9894 ± 0.0023 |
| Our model | random | 0.9999 ± 0.0000 | 0.9949 ± 0.0021 | 0.9957 ± 0.0017 | 0.9956 ± 0.0029 | 0.9957 ± 0.0018 | 0.9894 ± 0.0044 | 0.9894 ± 0.0044 |
| Paper model | spatial | 0.9991 ± 0.0009 | 0.9857 ± 0.0141 | 0.9987 ± 0.0011 | 0.9755 ± 0.0251 | 0.9868 ± 0.0132 | 0.9697 ± 0.0266 | 0.9691 ± 0.0277 |
| Our model | spatial | 0.9997 ± 0.0002 | 0.9900 ± 0.0054 | 0.9978 ± 0.0015 | 0.9844 ± 0.0116 | 0.9910 ± 0.0056 | 0.9779 ± 0.0108 | 0.9777 ± 0.0110 |

## 2. Susceptibility map: landslide density on held-out landslide samples (seed 0)

A good map has density rising with class. The paper's reported figure is 97.81% of landslides in High + Very High, covering 10.25% of the area.

| Model | Split | VeL area% / LD | Lo area% / LD | Mo area% / LD | Hi area% / LD | VeH area% / LD | Hi+VeH area | Hi+VeH landslides |
|---|---|---|---|---|---|---|---|---|
| Paper model | random | 35.0% / 0.00 | 8.2% / 0.00 | 6.4% / 0.07 | 8.2% / 0.05 | 42.2% / 2.35 | 50.4% | 99.6% |
| Our model | random | 24.3% / 0.00 | 18.6% / 0.01 | 18.7% / 0.05 | 17.1% / 0.61 | 21.3% / 4.15 | 38.4% | 98.9% |
| Paper model | spatial | 34.9% / 0.07 | 8.1% / 0.32 | 6.5% / 0.51 | 8.2% / 0.44 | 42.3% / 2.08 | 50.5% | 91.7% |
| Our model | spatial | 25.2% / 0.02 | 20.6% / 0.10 | 19.8% / 0.29 | 16.4% / 1.91 | 18.1% / 3.34 | 34.5% | 91.7% |

## 3. Reference: simple models on the raw factors (not part of the paper)

These show how easy the labels are: a plain logistic regression is already close to the top.

| Model | Split | AUC | OA | PRECISION | RECALL | F1 | MCC | KAPPA |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | random | 0.9981 ± 0.0004 | 0.9803 ± 0.0042 | 0.9841 ± 0.0042 | 0.9763 ± 0.0052 | 0.9802 ± 0.0042 | 0.9606 ± 0.0084 | 0.9606 ± 0.0084 |
| Random forest | random | 1.0000 ± 0.0000 | 0.9958 ± 0.0013 | 0.9978 ± 0.0014 | 0.9939 ± 0.0032 | 0.9958 ± 0.0013 | 0.9917 ± 0.0025 | 0.9917 ± 0.0026 |
| Logistic regression | spatial | 0.9949 ± 0.0028 | 0.9649 ± 0.0197 | 0.9824 ± 0.0120 | 0.9471 ± 0.0513 | 0.9635 ± 0.0255 | 0.9258 ± 0.0353 | 0.9238 ± 0.0384 |
| Random forest | spatial | 0.9993 ± 0.0006 | 0.9714 ± 0.0323 | 0.9901 ± 0.0098 | 0.9498 ± 0.0729 | 0.9679 ± 0.0386 | 0.9438 ± 0.0594 | 0.9408 ± 0.0648 |
