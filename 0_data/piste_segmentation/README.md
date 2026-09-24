# Piste Segmentation Data

This dataset was used to train the RF-DETR model for piste segmentation in Fencelytics.

## Dataset

The dataset contains **130 labeled images from 13 fencing videos**:

* **106 training images**
* **24 validation images**
* **10 images per video**

The videos were selected to provide variation in:

* Piste orientation
* Camera viewpoint
* Lighting conditions
* Scene composition

The segmentation model identifies the piste, which is then used to estimate its longitudinal direction and construct a piste-relative coordinate system for motion tracking.

## Data Availability

The dataset is provided for research reproducibility where redistribution is permitted. Source fencing footage may originate from publicly accessible broadcasts or online videos, but public accessibility does not necessarily grant redistribution rights.
