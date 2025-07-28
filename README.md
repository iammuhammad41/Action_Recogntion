# Action Recogntion
## Dataset:  HO-3D
### Accuracy: 96%
# Ho‑3D Action Recognition with Action Transformer

A PyTorch implementation for per‑frame action classification on the Ho‑3D dataset using a lightweight Transformer encoder.

## Features
- Parses and annotates Ho‑3D’s `object_pose.txt` & `action_class.txt`
- Balances classes via oversampling
- Scales pose features to [0,1]
- Trains an `nn.TransformerEncoder`‑based classifier
- Reports train/val loss & accuracy per epoch
- Outputs final test accuracy

## Requirements
- Python 3.7+
- PyTorch 1.7+
- scikit‑learn
- imbalanced‑learn
- pandas, numpy

## Installation
```bash
git clone https://github.com/you/ho3d-transformer.git
cd ho3d-transformer
pip install -r requirements.txt
