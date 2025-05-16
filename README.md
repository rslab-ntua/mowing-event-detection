#### Official implementation and dataset for the paper:
**Large Scale Mowing Event Detection on Dense Time Series Data Using Deep Learning Methods and Knowledge Distillation**

---

#### Abstract
The intensity of agricultural land use is a critical factor for food security and biodiversity preservation, necessitating effective
and scalable monitoring techniques. This study presents a novel approach for large-scale mowing event frequency detection us-
ing dense time series data and deep learning (DL) methods. Leveraging Sentinel-2 and Landsat data, we developed a bench-
mark dataset of over 1,600 annotated parcels in Greece, capturing mowing events through photo-interpretation and Enhanced
Vegetation Index (EVI) analysis. Four DL architectures were evaluated, including MLP, ResNet18, MLP+Transformer, and
Conv+Transformer, with additional handcrafted features incorporated to assess their impact on performance. Our results demon-
strate that the Conv+Transformer architecture achieved the highest improvement when enriched with additional features, while
ResNet18 showed a decline in performance under similar conditions. To address data scarcity, we employed knowledge distillation,
pre-training models on pseudo-labeled data derived from a dataset in Germany. This process significantly enhanced model perform-
ance, with fine-tuned ResNet18 and Conv+Transformer architectures achieving significant performance improvements. This study
highlights the importance of architecture selection, feature engineering, and pre-training strategies in time series classification for
agricultural monitoring. The proposed methods provide a scalable, non-invasive solution for monitoring mowing events, supporting
sustainable land management and compliance with agricultural policies. Future work will explore multimodal data integration and
advanced training techniques to further enhance detection accuracy.

---
### Data Preparation
The first step to train a model is to crop each of the parcels and keep them in the associated folder according to the number 
of events. The final folder structure must be like the following:
```
data
  |-- 0
  |-- 1
  |-- 2
  |-- 3
  |-- 4
```
