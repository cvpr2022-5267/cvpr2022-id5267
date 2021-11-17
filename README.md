# What Matters For Meta-Learning Vision Regression Tasks?
## Installation
- install CUDA10.1
- setup environment
```shell
  conda create -n wmmeta python=3.7 pip
  conda activate wmmeta
  pip install -r requirements.txt
```
## Datasets
download [dataset](https://drive.google.com/drive/u/2/folders/19I9FQ8LEoGffuVDq3opmkXpY_n7J35U4) under `data/`, datafolder should be structured as:
```shell
data/
├── distractor/
├── ShapeNet1D/
├── ShapeNet3D_azi180ele30/
dataset/
networks/
...
```

## Evaluation
evaluate and visualize predictions on distractor task:
```shell
python evaluate_and_plot_distractor.py --config cfg/evaluation/eval_and_plot/CNP_max_Distractor.yaml
```
evaluetate and visualize on ShapeNet1D:
```shell
python evaluate_and_plot_shapenet1d.py --config cfg/evaluation/eval_and_plot/ANP_ShapeNet1D.yaml
```
evaluate and visualize predictions on ShapeNet3D:
```shell
python evaluate_and_plot_shapenet3d.py --config cfg/evaluation/eval_and_plot/ANP_ShapeNet3D.yaml
```
Prediction vs context set size on novel tasks:
```shell
python evaluation.py --config cfg/evaluation/CNP_max_Distractor.yaml
python evaluation.py --config cfg/evaluation/ANP_ShapeNet1D.yaml
python evaluation.py --config cfg/evaluation/ANP_ShapeNet3D.yaml
```




