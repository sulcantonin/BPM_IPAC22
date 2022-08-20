# A Data-Driven Beam Trajectory Monitoring at the European XFEL
**Antonin Sulc, Raimund Kammering, Tim Wilksen (DESY, Hamburg)**

Interpretation of data from beam position monitors is a crucial part of the reliable operation of European XFEL. The interpretation of beam positions is often handled by a physical model, which can be prone to modeling errors or can lead to the high complexity of the computational model. In this paper, we show two data-driven approaches that provide insights into the operation of the SASE beamlines at European XFEL. We handle the analysis as a data-driven problem, separate it from physical peculiarities and experiment with available data based only on our empirical evidence and the data.

**Note (20/08/2022)**: As we mention in the paper, the hypersphere center ```c``` isn't kept fixed and is updated through the optimization, batching should cause slight variations of the ```c``` and prevent the collapse of all points projected on the ```c```. In the original paper, they particularly stress that this may lead to the projection of all points onto the ```c``` by zeroing all weights and keeping bias ```c = b```. In future experiments, we plan to stick to the suggested technique (i.e. remove biases and keep ```c``` fixed), since this explains why many interations cause diminishing scores. 

# Paper
- [paper](https://ipac2022.vrws.de/papers/mopopt069.pdf) 
- [poster](https://github.com/sulcantonin/BPM_IPAC22/blob/main/MOPOPT069_poster.pdf)
- [code](model/code.py)
- [torch sase13](model14_sa13_10iter_dropout0.1/model_009.torch) 
- [torch sase2](model14_sa2_10iter_dropout0.1/model_009.torch)

# Data

The files are compressed into pickles with gzip package with 8-bit float precission (for storage purposes). The pickle contains dictioary with ```first_bunch_x``` and ```first_bunch_y``` for horizontal and vertical BPMs respectively. Each element is a pandas (pandas 1.3.1) data frame. Each row is position of first bunch in a single indidivual bunch train.

## Reported Issues
Two issues shown in the paper. 

### SASE1 undulator server crashed after an unusual selection of colours for individual cells (2022-03-07 20:45:00)

![image](https://raw.githubusercontent.com/sulcantonin/BPM_IPAC22/bc8be7f2adfa2267333bc138d6728f29d1eaf8e2/public_data/undulator.png)
[zip](public_data/undulator.zip).
### A phase shifter at SASE3 does not move (2022-04-25 21:23:42)
![image](https://raw.githubusercontent.com/sulcantonin/BPM_IPAC22/bc8be7f2adfa2267333bc138d6728f29d1eaf8e2/public_data/phase_shifter.png)
[zip](public_data/phase_shifter.zip)

# Model
To understand how to read data, check ```get_X_from_file``` in ```model.py```. The function loads a list of files and normalizes it at the same time. 

The model requires following libraries:
- pytorch '1.10.2+cu113'
- numpy '1.20.3'
- pandas '1.3.1'

