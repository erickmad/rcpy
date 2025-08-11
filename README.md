# Reservoir Computing with Reservoirpy

## Structure of project

```
rcpy/
│
├── rcpy/
│ ├── __init__.py
│ ├── data/
│ │ ├── __init__.py
│ │ ├── data_retrieval.py
│ │ └── utils_data.py
│ └── forecasting/
│ ├── __init__.py
│ ├── forecasting_rcpy.py
│ └── utils_forecasting_rcpy.py
│
├── notebooks/
│ └── demo_notebook.ipynb
│
├── setup.py
├──  requirements.txt
└── README.md
```



## Questions

- data > utils_data_rcpy > preprocess_data_rcpy
  
  - Do I want to normalize per variable?
  
  - Should I keep `init_transient` and `transient_length` or should it just be one? <-- does this have to do with the transient when training the outputs?

- forecasting > forecasting_rcpy > `forecast_rcpy`
  
  - Should I reset the internal states when starting prediction?
