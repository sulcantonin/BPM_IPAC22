# A Data-Driven Beam Trajectory Monitoring at the European XFEL
** Antonin Sulc, Raimund Kammering, Tim Wilksen (DESY, Hamburg) **

Interpretation of data from beam position monitors is a crucial part of the reliable operation of European XFEL. The interpretation of beam positions is often handled by a physical model, which can be prone to modeling errors or can lead to the high complexity of the computational model. In this paper, we show two data-driven approaches that provide insights into the operation of the SASE beamlines at European XFEL. We handle the analysis as a data-driven problem, separate it from physical peculiarities and experiment with available data based only on our empirical evidence and the data.

# Paper
[paper](https://ipac2022.vrws.de/papers/mopopt069.pdf) [poster](https://github.com/sulcantonin/BPM_IPAC22/blob/main/MOPOPT069_poster.pdf)

# Data
## Reported Issues in logbook input data
The files are compressed into pickles with gzip packages with 8-bit float precission. The pickle contains dictioary with 'first_bunch_x' and 'first_bunch_y' for horizontal and vertical BPMs respectively where each element is a pandas (pandas 1.3.1) data frame. Each row is position of first bunch in a single indidivual bunch train and column a BPM.

### SASE1 undulator server crashed after an unusual selection of colours for individual cells (2022-03-07 20:45:00)
[TBD](TBD).
### A phase shifter at SASE3 does not move (2022-04-25 21:23:42)
# Source files
[TBD](TBD)

# Model
## Model Source Code 
[TBD](TBD)
## Model Trained Weights
[TBD](TBD)
