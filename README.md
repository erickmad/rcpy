# Reservoir Computing with Reservoirpy

## Description

Code to perform time series forecasts using Echo State Networks: building the reservoir, training the readout layer adn performing the forecasts in iterative mode.

## Installation

On the project's root folder: run

```shell
pip install -e .
```



## Structure of project

```shell
rcpy/
│
├── src/
│   └── rcpy/
│      ├── analysis/
│      ├── data/
│      ├── enso/
│      ├── forecasting/
│      ├── hypopt/
│      ├── minimal_esn/
│      ├── models/
│      ├── plotting/
│      ├── setup_experiments/
│      ├── training/
│      └── utilities/
│
├── notebooks/
│   └── demo_notebooks.ipynb
│
├── environment.yml
├── setup.py
├── requirements.txt
└── README.md
```

## Questions

- data > utils_data_rcpy > preprocess_data_rcpy
  
  - Do I want to normalize per variable?
  
  - Should I keep `init_transient` and `transient_length` or should it just be one? <-- does this have to do with the transient when training the outputs?

- forecasting > forecasting_rcpy > `forecast_rcpy`
  
  - Should I reset the internal states when starting prediction?
