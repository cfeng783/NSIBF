## Getting Started

#### Install dependencies (with python 3.6) 

```shell
pip install -r requirements.txt
```

#### Run the code

```shell
cd experiments
python main.py
```

If you want to change the default configuration, you can edit `ExpConfig` in `main.py` or overwrite the config in `main.py` using command line args. For example:

```
python main.py --dataset='MSL' --max_epoch=20
```
