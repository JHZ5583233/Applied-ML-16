# Applied ML Team 16

Welcome to our repo for our project for Applied machine learning project.

Our project is to train an model to approximate depth data from rgb images in a tilling pattern.

## Prerequisites
Make sure you have the following software and tools installed:

- **Conda**: Conda is used for dependency management. This tools is a standard for a lot of machine learning libraries and has support for pip packages as well. "conda install --yes --file conda_requirements.txt"
- **python 3.11**: a tested version of python that this repository works in.

## Getting Started
### general
1. clone this repository.
2. get the training and validation data and place in the data folder in the project_name folder.
    * Data used to train our model
        1. download from google drive. [drive](https://drive.google.com/drive/folders/1k_WsVOjaULgb3N2JebxjTVqJjVsw85dP)
    * original data
        1. download from original data. [website](https://diode-dataset.org/)
        2. run subset_maker.py with amount sample and "val or "train" to get workable data folder.
3. create conda environment with python 3.11
4. instal the packages from "conda_requirements.txt" using the command under.
```basg
conda install --yes --file conda_requirements.txt"
```
5. install pytorch using the command from their [website](https://pytorch.org/).
6. download the model from the release page and place it in the root folder of the repository

### Train and validate
Both the commands below should be ran in the root folder of the repository

* #### Train
```bash
python main.py --epochs (amount epoch) --batch-size (batch size) --lr (learning rate) --freeze-epochs (amount before freeze)
```

* #### Validate
```bash
python main.py evaluate (model file name with extension) --batch-size (batch size)
```

### Train and validate
Both the commands below should be ran in the root folder of the repository

#### Train
1. python main.py --epochs (amount epoch) --batch-size (batch size) --lr (learning rate) --freeze-epochs (amount before freeze)

#### Validate
1. python main.py evaluate (model file name with extension) --batch-size (batch size)

### API
1. run the following command in the root directory of the repository.
```bash
uvicorn FastAPI:app --reload"
```

### Streamlit
1. run the following command in the root directory of the repository.
```bash
streamlit run streamlit_main.py" when in the main folder of repository.
```
2. follow instruction on the web demo.

### Unit testing

To run all the tests developed using _unittest_, simply use:
```bash
python -m unittest discover tests
```
If you wish to see additional details, run it in verbose mode:
```bash
python -m unittest discover -v tests
```

repository map:
```bash
├───.github
│   └────workflows
│        └──── style.yml
├───project_name
│   ├───data
│   │   ├───val_subset # only there when validating
│   │   ├───train_subset # only there when training
│   │   ├───data_loader.py
│   │   ├───data_test.py
│   │   ├───path_grapper.py
│   │   └───subset_maker.py
│   ├───models
│   │   ├───cnn.py
│   │   └───Preprocessing_class.py
│   └───Training
│       ├───Evaluation
│       │   ├───evaluate.py
│       │   └───validation.py
│       └───model_trainer.py
├───tests
│   ├───data
│   ├───features
│   └───models
├───.gitignore
├───cnn_best.pth # download from release page
├───main.py
├───streamlit_main.py
├───FastAPI.py
├───conda_requirements.txt
└───README.md
```