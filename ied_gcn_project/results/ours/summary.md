# Our model (paper + one-hot factors + logit classing)

Mean ± std over seeds.

| Split | Runs | AUC | OA | PRECISION | RECALL | F1 | MCC | KAPPA |
|---|---|---|---|---|---|---|---|---|
| random | 5 | 0.9999 ± 0.0000 | 0.9949 ± 0.0021 | 0.9957 ± 0.0017 | 0.9956 ± 0.0029 | 0.9957 ± 0.0018 | 0.9894 ± 0.0044 | 0.9894 ± 0.0044 |
| spatial | 5 | 0.9997 ± 0.0002 | 0.9900 ± 0.0054 | 0.9978 ± 0.0015 | 0.9844 ± 0.0116 | 0.9910 ± 0.0056 | 0.9779 ± 0.0108 | 0.9777 ± 0.0110 |

### Landslide density, all landslide samples (random split, seed 0)

| Class | PC (area share) | PL (landslide share) | LD |
|---|---|---|---|
| VeL | 0.2434 | 0.0000 | 0.00 |
| Lo | 0.1858 | 0.0007 | 0.00 |
| Mo | 0.1871 | 0.0138 | 0.07 |
| Hi | 0.1706 | 0.1005 | 0.59 |
| VeH | 0.2130 | 0.8851 | 4.16 |

### Landslide density, held-out landslide samples (random split, seed 0)

| Class | PC (area share) | PL (landslide share) | LD |
|---|---|---|---|
| VeL | 0.2434 | 0.0000 | 0.00 |
| Lo | 0.1858 | 0.0021 | 0.01 |
| Mo | 0.1871 | 0.0086 | 0.05 |
| Hi | 0.1706 | 0.1049 | 0.61 |
| VeH | 0.2130 | 0.8844 | 4.15 |

### Landslide density, all landslide samples (spatial split, seed 0)

| Class | PC (area share) | PL (landslide share) | LD |
|---|---|---|---|
| VeL | 0.2516 | 0.0013 | 0.01 |
| Lo | 0.2057 | 0.0059 | 0.03 |
| Mo | 0.1978 | 0.0263 | 0.13 |
| Hi | 0.1640 | 0.1477 | 0.90 |
| VeH | 0.1808 | 0.8188 | 4.53 |

### Landslide density, held-out landslide samples (spatial split, seed 0)

| Class | PC (area share) | PL (landslide share) | LD |
|---|---|---|---|
| VeL | 0.2516 | 0.0048 | 0.02 |
| Lo | 0.2057 | 0.0214 | 0.10 |
| Mo | 0.1978 | 0.0570 | 0.29 |
| Hi | 0.1640 | 0.3135 | 1.91 |
| VeH | 0.1808 | 0.6033 | 3.34 |
