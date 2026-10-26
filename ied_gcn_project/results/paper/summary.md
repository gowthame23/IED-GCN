# Paper model (IED-GCN as published)

Mean ± std over seeds.

| Split | Runs | AUC | OA | PRECISION | RECALL | F1 | MCC | KAPPA |
|---|---|---|---|---|---|---|---|---|
| random | 5 | 0.9998 ± 0.0001 | 0.9949 ± 0.0011 | 0.9952 ± 0.0015 | 0.9961 ± 0.0029 | 0.9957 ± 0.0010 | 0.9894 ± 0.0023 | 0.9894 ± 0.0023 |
| spatial | 5 | 0.9991 ± 0.0009 | 0.9857 ± 0.0141 | 0.9987 ± 0.0011 | 0.9755 ± 0.0251 | 0.9868 ± 0.0132 | 0.9697 ± 0.0266 | 0.9691 ± 0.0277 |

### Landslide density, all landslide samples (random split, seed 0)

| Class | PC (area share) | PL (landslide share) | LD |
|---|---|---|---|
| VeL | 0.3502 | 0.0000 | 0.00 |
| Lo | 0.0819 | 0.0007 | 0.01 |
| Mo | 0.0640 | 0.0046 | 0.07 |
| Hi | 0.0824 | 0.0046 | 0.06 |
| VeH | 0.4215 | 0.9902 | 2.35 |

### Landslide density, held-out landslide samples (random split, seed 0)

| Class | PC (area share) | PL (landslide share) | LD |
|---|---|---|---|
| VeL | 0.3502 | 0.0000 | 0.00 |
| Lo | 0.0819 | 0.0000 | 0.00 |
| Mo | 0.0640 | 0.0043 | 0.07 |
| Hi | 0.0824 | 0.0043 | 0.05 |
| VeH | 0.4215 | 0.9914 | 2.35 |

### Landslide density, all landslide samples (spatial split, seed 0)

| Class | PC (area share) | PL (landslide share) | LD |
|---|---|---|---|
| VeL | 0.3490 | 0.0072 | 0.02 |
| Lo | 0.0807 | 0.0079 | 0.10 |
| Mo | 0.0652 | 0.0098 | 0.15 |
| Hi | 0.0819 | 0.0138 | 0.17 |
| VeH | 0.4232 | 0.9613 | 2.27 |

### Landslide density, held-out landslide samples (spatial split, seed 0)

| Class | PC (area share) | PL (landslide share) | LD |
|---|---|---|---|
| VeL | 0.3490 | 0.0238 | 0.07 |
| Lo | 0.0807 | 0.0261 | 0.32 |
| Mo | 0.0652 | 0.0333 | 0.51 |
| Hi | 0.0819 | 0.0356 | 0.44 |
| VeH | 0.4232 | 0.8812 | 2.08 |
