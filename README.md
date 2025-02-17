# CXR Dataset Pipeline

This repository contains models and code for preprocessing chest X-ray (CXR) images. The pipeline includes several key preprocessing steps to ensure data quality and consistency:

## Features

- Out-of-Distribution (OOD) Detection
- Monochrome Image Detection
- View Position Classification (AP/PA/LAT)

## Getting Started
Download the pre-trained model weights from [Google Drive](https://drive.google.com/file/d/17VfzcZtna5bfTSR-tv6MLDHJiwCUlMGr/view?usp=sharing)


### Prerequisites
- Python 3.10
- Additional requirements listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/lukeingawesome/CXR_dataset_pipeline.git
conda create -n cxr python=3.10
cd CXR_dataset_pipeline
conda activate cxr
pip install -r requirements.txt
```
