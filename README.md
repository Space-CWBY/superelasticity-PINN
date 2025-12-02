# superelasticity-PINN

## Project overview
- Python source code for training and evaluating a physics-informed neural network model for superelastic constitutive behavior  
- Boundary condition files for constructing physics-informed constraints on ξ and εₚ


This repository provides a cleaned version of the final research code.

## Model
```
/model
```

Contains the core implementation of the model:

- scripts to train the network on processed stress–strain data  
- evaluation scripts for generating model predictions  


## Data
```
/data
```

Contains boundary condition files in .csv format to:

- build inequality/equality constraints for ξ and εₚ during training


## Installation
Python 3.10 or newer is recommended.


## Citation
If you use this code or model in your research, please cite our work  
(citation details to be added upon publication).
