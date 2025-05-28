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
2. instal the packages from "conda_requirements.txt"
3. download the model from the release page and place it in the root folder of the repository

### Train and validate

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
│   └────data
│        └──── style.yml
├───project_name
│   ├───data  # For data processing, not storing .csv
│   ├───features
│   └───models  # For model creation, not storing .pkl
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