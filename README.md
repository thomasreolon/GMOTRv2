# GMOTRv2: End-to-End General Multi-Object Tracking

## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [MOTRv2](https://github.com/megvii-research/MOTRv2).

### Requirements

* Install pytorch using conda (optional)

    ```bash
    conda create -n venv python=3.7
    conda activate venv
    conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
    ```

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Train

* Set Up
    - download the datasets (FSCD-147, COCO17, GMOT-40, Synth, ...)
    - and set their paths in configs/_paths.args

* Launch training & testing
    ```bash
    bash train.sh configs/m.original.args
    bash test.sh
    ```


## Acknowledgements

- [MOTRv2](https://github.com/megvii-research/MOTRv2)