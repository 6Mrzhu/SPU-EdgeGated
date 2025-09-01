
## Data Preparation

We use two public datasets, including PU1K ([Paper](https://arxiv.org/abs/1912.03264), [Download](https://drive.google.com/file/d/1oTAx34YNbL6GDwHYL2qqvjmYtTVWcELg/view?usp=sharing))

## Usage

The code is tested under Pytorch 1.6.0 and Python 3.6 on Ubuntu 20.04. Pretrained weights are available in ` ./model/release `. 

1. Install python denpendencies.

```shell
pip install -r requirements.txt
```

#### Install KNN_cuda
```shell
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
2. Compile pyTorch extensions.

```shell
cd pointnet2_ops_lib
python setup.py install


cd pointops
python setup.py install

cd ../losses
python setup.py install
```

3. Train the model. 

```shell
sh start_train.sh
```

4. Evaluate the model.

```shell
cd evaluation_code
cmake .
make

sh test.sh
```

## License

This project is released under the [Apache 2.0 license](./LICENSE). Other codes from open source repository follows the original distributive licenses.


## Acknowledgement

We borrow some code from [MPU](https://github.com/yifita/3PU_pytorch), [PU-Net](https://github.com/yulequan/PU-Net), [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet), and [PUGCN](https://github.com/guochengqian/PU-GCN). 


## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.



```
