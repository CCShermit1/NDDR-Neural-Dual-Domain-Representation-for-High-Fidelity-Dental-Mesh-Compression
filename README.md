# **NDDR: Neural Dual-Domain Representation for High-Fidelity Dental Mesh Compression**


![](resources/render+color.png)

# Usage

## Preparation

### third-party:
```
draco: https://github.com/google/draco/tree/7d58126d076bc3f5f9d8c114d1700b7311faecfe
glm: https://github.com/g-truc/glm/tree/5c46b9c07008ae65cb81ab79cd677ecc1934b903
imgui: https://github.com/ocornut/imgui/tree/d3c3514a59bb31406c954c2b525f330e9d167845
implot: https://github.com/epezent/implot/tree/f156599faefe316f7dd20fe6c783bf87c8bb6fd9
littlevk: https://github.com/iveevi/littlevk/tree/da008ec4d4e573a3aa8c92cc9f57802cc040af93
nvdiffmodeling: https://github.com/NVlabs/nvdiffmodeling/tree/9b2ba2eff83c7d90127f78c20773b06ddc3ae1db
stb: https://github.com/nothings/stb/tree/ae721c50eaf761660b4f90cc590453cdb0c2acd0
```

## Enviroments
```
python = 3.10
torch = 2.4.0
pip install -r requirements.txt
```

## Run
run `python source/train.py` on your data

```
train.py [--mesh MESH] [--lod LOD] [--features FEATURES] [--display DISPLAY] [--batch BATCH] [--fixed-seed]

options:
  --mesh MESH          Target mesh
  --lod LOD            Number of patches to partition
  --features FEATURES  Feature vector size
  --display DISPLAY    Display the result after training
  --batch BATCH        Batch size for training
  --fixed-seed         Fixed random seed (for debugging)
```
The results will be stored in the `results` folder