# A Data-Driven Beam Trajectory Monitoring at the European XFEL
**Antonin Sulc, Raimund Kammering, Tim Wilksen (DESY, Hamburg)**

Interpretation of data from beam position monitors is a crucial part of the reliable operation of European XFEL. The interpretation of beam positions is often handled by a physical model, which can be prone to modeling errors or can lead to the high complexity of the computational model. In this paper, we show two data-driven approaches that provide insights into the operation of the SASE beamlines at European XFEL. We handle the analysis as a data-driven problem, separate it from physical peculiarities and experiment with available data based only on our empirical evidence and the data.

# Paper
- [paper](https://ipac2022.vrws.de/papers/mopopt069.pdf) 
- [poster](https://github.com/sulcantonin/BPM_IPAC22/blob/main/MOPOPT069_poster.pdf)

# Data

The files are compressed into pickles with gzip package with 8-bit float precission (for storage purposes). The pickle contains dictioary with ```first_bunch_x``` and ```first_bunch_y``` for horizontal and vertical BPMs respectively. Each element is a pandas (pandas 1.3.1) data frame. Each row is position of first bunch in a single indidivual bunch train.

## Reported Issues
Two issues shown in the paper. 

### SASE1 undulator server crashed after an unusual selection of colours for individual cells (2022-03-07 20:45:00)
[zip](public_data/undulator.zip).
### A phase shifter at SASE3 does not move (2022-04-25 21:23:42)
# Source files
[zip](public_data/phase_shifter.zip)

# Model
To understand how to read data, check ```get_X_from_file``` in ```model.py```. The function loads a list of files and normalizes it at the same time. 

The model requires following libraries
- pytorch '1.10.2+cu113'
- numpy '1.20.3'
- pandas tbd
## Model Source Code 
[code](model/code.py)
## Model Trained Weights
[folder sase 1,3](model14_sa13_10iter_dropout0.1/) [folder sase 2](model14_sa2_10iter_dropout0.1/)
