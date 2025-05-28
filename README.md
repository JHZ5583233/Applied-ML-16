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
2. create conda environment with python 3.11
3. instal the packages from "conda_requirements.txt"
4. download the model from the release page and place it in the root folder of the repository

### Train and validate
Both the commands below should be ran in the root folder of the repository

#### Train
1. python main.py --epochs (amount epoch) --batch-size (batch size) --lr (learning rate) --freeze-epochs (amount before freeze)

#### Validate
1. python main.py evaluate (model file name with extension) --batch-size (batch size)

### API
1. run "uvicorn FastAPI:app --reload"

### Streamlit
1. run "streamlit run streamlit_main.py" when in the main folder of repository.
2. follow instruction on the web demo.

### Unit testing
You are expected to test your code using unit testing, which is a technique where small individual components of your code are tested in isolation.

An **example** is given in _tests/test_main.py_, which uses the standard _unittest_ Python module to test whether the function _hello_world_ from _main.py_ works as expected.

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
├───main.py
├───streamlit_main.py
├───FastAPI.py
├───conda_requirements.txt
└───README.md
```