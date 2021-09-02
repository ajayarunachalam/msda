## Overview 

pypi: https://pypi.org/project/msda/

Related Blogs: https://towardsdatascience.com/explainable-ai-xai-design-for-unsupervised-deep-anomaly-detector-6bd1275ed3fc

               https://ajay-arunachalam08.medium.com/multi-dimensional-time-series-data-analysis-unsupervised-feature-selection-with-msda-package-430900a3829a

MSDA is a prototype for unsupervised feature selection and/or unsupervised deep convolutional neural network & lstm autoencoders based real-time anomaly detection from high-dimensional heterogeneous/homogeneous time series multi-sensor data. It also includes a module of Explainable AI for the built time-series anomaly predictor.

Intuitive representation of the unsupervised feature selection is as shown below.

![alt text](https://github.com/ajayarunachalam/msda/blob/main/conceptual_framework_msda_new.png)

Intuitive representation of the unsupervised real-time point anomalies detection is as shown below.

![alt text](https://github.com/ajayarunachalam/msda/blob/main/anomalies_msda.png)

From local explanations to global understanding with explainable AI for trees - motivation from here - https://www.nature.com/articles/s42256-019-0138-9

![alt text](https://github.com/ajayarunachalam/msda/blob/main/shap_conceptual.png), Image credits - https://github.com/slundberg/shap

# MSDA 1.10.0

## What is MDSA?
MSDA is an open source `low-code` Multi-Sensor Data Analysis library in Python that aims to reduce the hypothesis to insights cycle time in a time-series multi-sensor data analysis & experiments. It enables users to perform end-to-end proof-of-concept experiments quickly and efficiently. The module identifies events in the multidimensional time series by capturing the variation and trend to establish relationship aimed towards identifying the correlated features helping in feature selection from raw sensor signals. Also, it provides a provision to precisely detect the anomalies in real-time streaming data an unsupervised deep convolutional neural network & also a lstm autoencoders based detectors are designed to run on GPU/CPU. Finally, a game theoretic approach is used to explain the output of the built anomaly detector model. 


The package includes:-
1) Time series analysis.
2) The variation of each sensor column wrt time (increasing, decreasing, equal).
3) How each column values varies wrt other column, and the maximum variation ratio between each column wrt other column.
4) Relationship establishment with trend array to identify most appropriate sensor.
5) User can select window length and then check average value and standard deviation across each window for each sensor column.
6) It provides count of growth/decay value for each sensor column values above or below a threshold value.
7) Feature Engineering 
    a) Features involving trend of values across various aggregation windows: change and rate of change in average, std. deviation across window.
    b) Ratio of changes, growth rate with std. deviation.
    c) Change over time.
    d) Rate of change over time.
    e) Growth or decay.
    f) Rate of growth or decay.
    g) Count of values above or below a threshold value.
8) ** Unsupervised deep time-series anomaly detector. **
9) ** Game theoretic approach to explain the time-series data model. **


MSDA is `simple`, `easy to use` and `low-code`. 

## Features

![alt text](https://github.com/ajayarunachalam/msda/blob/main/features_msda_new.png)

## Unsupervised FS Workflow

![alt text](https://github.com/ajayarunachalam/msda/blob/main/flowchart_msda.png)

## Unsupervised time-series anomaly detector workflow

** Deep Convolutional Neural Network **

![alt text](https://github.com/ajayarunachalam/msda/blob/main/deepCNN.gif) inspiration from this IEEE paper - https://ieeexplore.ieee.org/document/8581424

** LSTM Autoencoder **

![alt text](https://github.com/ajayarunachalam/msda/blob/main/lstm_ae.png) inspiration from here - https://www.nature.com/articles/s41598-019-55320-6

## Features Coming Soon***

1) Explainable Forecasting.
2) ACF/PACF Analysis.
3) Detection of False Trading Strategies Using Deep Unsupervised/Reinforcement Learning Methods.
4) Optimization of the Trading Strategies (Long & Short Term) to maximize profit decision making.
4) 3D Distribution Maps for MOX gas sensor signals.

## Installation
The easiest way to install msda is using pip. 

```python
pip install msda
```
```terminal 
$ git clone https://github.com/ajayarunachalam/msda
$ cd msda
$ python setup.py install
```

## Notebook
```notebook
!pip install msda
```
Follow the rest as demonstrated in the demo example for Unsupervised Feature Selection [here] -- https://github.com/ajayarunachalam/msda/blob/main/demo.ipynb

Follow the rest as demonstrated in the demo example for Unsupervised Deep Anomaly Detectors & Time series predictor as Explainable AI [here] -- https://github.com/ajayarunachalam/msda/blob/main/demo1_v1.ipynb

## Dependencies
Most of the dependencies are installed automatically. But, if not installed when you install MSDA, then these dependencies must be installed as shown below.

```shell
pip install pandas
pip install numpy
pip install matplotlib
pip install datetime
pip install statistics
pip install torch
pip install seaborn
pip install sklearn
pip install scipy
pip install shap
pip install keras
pip install ipywidgets
```

## Python:
Installation is only supported on 64-bit version of Python. Tested on numpy version '1.18.3', pandas <= '1.0.5', torch == 1.4.0, keras == 2.0.8, seaborn = '0.9.0', shap = '0.39.0', ipywidgets == 7.5.1

## Important Links
- Example Unsupervised Feature Selection Demo Notebook : https://github.com/ajayarunachalam/msda/blob/main/demo.ipynb
- Example Unsupervised Anomaly Detector & Explainable AI Demo Notebook : https://github.com/ajayarunachalam/msda/blob/main/demo1_v1.ipynb


## Who should use MSDA?
MSDA is an open source library that anybody can use. In our view, the ideal target audience of MSDA is: <br />

- Researchers for quick poc testing.
- Experienced Data Scientists who want to increase productivity.
- Citizen Data Scientists who prefer a low code solution.
- Students of Data Science.
- Data Science Professionals and Consultants involved in building Proof of Concept projects.



## License

Copyright 2021 Ajay Arunachalam <ajay.arunachalam08@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
© 2021 GitHub, Inc.
