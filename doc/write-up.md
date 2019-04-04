Alastair Hamilton, Spring 2019
# Neural Nets for Structural Dynamic Response Estimation
This project is inspired by [Wu and Jahanshahi's](./2018 - Deep Convolutional Neural Network for Structural Dynamic Response Estimation and System Identification.pdf) work with using CNN's for estimating the dynamic response of structures. I wanted to study what they did and how neural nets were useful for the kind of regression they were doing in order to benefit my own understanding of the field and to see if I could make it better.

## Contents
- Problem description
- Data
 - Batch
 - Walk-forward
 - Walk-forward with previous data
- Error calculations
- Architectures
 - DNN
 - CNN
 - LSTM
- Results
- Evaluation
- Conclusion

## Problem
In this work I looked at the simple linear single-degree of freedom (SDOF) example and tried to estimate 2 cases:
1. the displacement given some excitation, acceleration and velocity and
2. the acceleration given some excitation only.

It was also important to consider how noise in the excitation affected the results and so 5%, 10%, 20% and 30% noise cases were investigated.

### Linear SDOF




### Data
#### Batch
Produced discontiunities in predictions
#### Walk-forward
Useful for generating more data from limited data
#### Walk-forward with more previous data
Dependance on past data so requires past data before predicting current data. Should also help discontinuities.
