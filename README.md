#### Official implementation and dataset for the paper:
**Large Scale Mowing Event Detection on Dense Time Series Data Using Deep Learning Methods and Knowledge Distillation**

---

:rocket: **Actively Maintained**  
:test_tube: **Experimental code may be present**

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

### Harmonised Mowing Event Dataset

The dataset developed for this study is publicly available and currently under continuous improvement. This dataset consists of 3 areas in Greece in the 
form of precomputed EVI timeseries and 3 geojson files which contain the number of mowing events.



It can be downloaded from here.

---
### Data Preparation
The first step to train a model is to crop each of the parcels and keep them in the associated folder according to the number 
of events. The final folder structure must be like the following:
```
└── data
    ├── 0
    ├── 1
    ├── 2
    ├── 3
    └── 4
```
The proposed code to perform the above is the following:

```python
import fiona
import rasterio
import rasterio.mask
import numpy as np
import os
import json
from tqdm import tqdm


# input folder for the timeseries data to crop
input_folder = <folder for input ts>


## export folder for the dataset##
##each different class--> different folder##
export_folder_path = <folder for output ts>

INDEX_DESCRTPTION = '_full_hls'

area_dict = {'17':[0,1,2,3,4]}


for i,key in enumerate(area_dict):
    print('area is',key)
    values = area_dict[key]
    print('values are',values)
    for j,value in enumerate(values):
        area = key
        target_attribute_value = value
        shapefile_path = f'./mowing_annotation_{area}.shp'

        with fiona.open(shapefile_path, "r") as shapefile:
            selected_shapes = [feature["geometry"] for feature in shapefile if feature["properties"]["count"] == target_attribute_value]
            #print(selected_shapes)
       
        for k,shape in enumerate(selected_shapes):
            with rasterio.open(input_folder+f'kampos_{area}_image_export{INDEX_DESCRTPTION}.tif') as src:
                # Mask the raster based on the selected shapefile geometries
                out_image, out_transform = rasterio.mask.mask(src, [shape], crop=True, nodata=np.nan)
                out_meta = src.meta  # Get metadata from the source raster
                
            
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
            output_raster_path = export_folder_path+str(target_attribute_value)+'/'+str(i)+str(j)+str(k)+'.tif'
            with rasterio.open(output_raster_path, "w", **out_meta) as dest:
                dest.write(out_image)
```
## Dataset download
🗂️ Data is available from the paper's official [data upload](https://zenodo.org/records/15442875). 

---
### Environment preparation

A requirements.txt file is provided to be used with conda. We recomment python>=3.10 and a fresh environment using the provided file. 

---

### Training

To train an MLP model which is the recomended for start, follow the main.ipynb

---

### How to cite

Moumouris, T., Tsironis, V., Psalta, A., and Karantzalos, K.: Large Scale Mowing Event Detection on Dense Time Series Data Using Deep Learning Methods and Knowledge Distillation, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLVIII-M-7-2025, 43–48, https://doi.org/10.5194/isprs-archives-XLVIII-M-7-2025-43-2025, 2025

----

