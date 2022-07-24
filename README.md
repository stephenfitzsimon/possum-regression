# Possum Regression and Classification Model
*Predicting the sex and age of the Short Eared and Mountain Bushtail possums*

## Key Takeaways
- The ratio of various head measurements to various body measurements is different between males and females; however, the anatomical measurements themselves did not differ between the sexes.
- The classification model for the sex classification is ~0.97 accurate
    - This high accuracy is most likely an accident of the sample represented by the dataset

## What I Learned
- Subplotting using seaborn
- Classification models using scikit learn

## Information in this repository

The analysis of the data and the models can be found <a href='https://github.com/stephenfitzsimon/possum-regression/blob/main/possum_models.ipynb'>here.</a>

Information on the data can be found <a href='https://www.kaggle.com/datasets/abrambeyer/openintro-possum'>here.</a>

`possum_model.ipynb` will contain the final analysis and models.

`acquire.py` contains all the code needed to acquire and clean the dataframe for the analysis and modelling

`possum.csv` contains the data as it was posted on kaggle as of 8 July 2022

## Reproducing this project

1. Download this repository
2. Run `possum_model.ipynb`