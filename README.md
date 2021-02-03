![alt text](https://github.com/ajayarunachalam/msda/blob/master/conceptual_framework_msda.png)

# MSDA 1.0.0

## What is MDSA?
MSDA is an open source `low-code` Multi-Sensor Data Analysis library in Python that aims to reduce the hypothesis to insights cycle time in a time-series multi-sensor data analysis & experiments. It enables users to perform end-to-end proof-of-concept experiments quickly and efficiently. The module identifies events in the multidimensional time series by capturing the variation and trend to establish relationship aimed towards identifying the correlated features helping in feature selection from raw sensor signals.


The package includes:-
1) Time series analysis
2) The variation of each sensor column wrt time (increasing, decreasing, equal)
3) How each column values varies wrt other column, and the maximum variation ratio between each column wrt other column
4) Relationship establishment with trend array to identify most appropriate sensor
5) User can select window length and then check average value and standard deviation across each window for each sensor column
6) It provides count of growth/decay value for each sensor column values above or below a threshold value
7) Feature Engineering 
    a) Features involving trend of values across various aggregation windows: change and rate of change in average, std. deviation across window
    b) Ratio of changes, growth rate with std. deviation
    c) Change over time
    d) Rate of change over time
    e) Growth or decay
    f) Rate of growth or decay
    g) Count of values above or below a threshold value 


MSDA is `simple`, `easy to use` and `deployment ready`. 

## Features

![alt text](https://github.com/ajayarunachalam/msda/blob/master/features_msda.png)

## Workflow

![alt text](https://github.com/ajayarunachalam/msda/blob/master/flowchart_msda.png)

## Installation
The easiest way to install pycaret is using pip. 

```python
pip install msda
```
```terminal 
$ git clone https://github.com/ajayarunachalam/msda
$ cd msda
$ python setup.py install
```

## Dependencies
Most of the dependencies are installed automatically. But, if not installed when you install MSDA, then these dependencies must be installed as shown below.

```shell
pip install pandas
pip install numpy
pip install matplotlib
pip install IPython
pip install ipywidgets
pip install datetime
pip install statistics
```

## Python:
Installation is only supported on 64-bit version of Python.

## Important Links
- Example Demo Notebook : https://github.com/ajayarunachalam/msda/tree/master/demo.ipynb


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
