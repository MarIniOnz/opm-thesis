# Master Thesis on Characterization and Classification of Finger Movement-Related Brain Activity using Optically Pumped Magnetometers.

This repository contains the publicly available code associated with the Master's thesis project by [Martin Iniguez de Onzono](https://github.com/MarIniOnz/), conducted in the Clinical Neurotechnology Lab at Charité – Universitätsmedizin Berlin. 

Thesis supervised by Prof. Dr. med. Surjo R. Soekadar, Dr. Livia De Hoz, and [Jan Zerfowski, MSc](https://github.com/jzerfowski/).

## Content of the repository
- `readme.md`: Readme file containing the structure of the repository.
- `Makefile`: File containing a set of directives used with the *make* command line utility to automate the build process of a project. It includes commands for compiling code, installing dependencies, running tests, and formatting the files in the repository.
- `readme.md`: Readme file containing the structure of the repository.
- `pyproject.toml`: This file specifies the build system requirements for the Python project. It is used to define settings such as the package metadata, dependencies, and compatible versions.
- `setup.py`: A setup script for the package. This particular script specifies that the package is named *opm_thesis* and includes all sub-packages within the *opm_thesis* directory.
- `requirements.txt`: This file lists the dependencies for the Python project, specifying exact versions to ensure consistent environments.
- `master_thesis.pdf`: The Master Thesis pdf-file (this one).
- `notebooks`: Folder containing the notebooks that created the figures and tables of thesis' Results section.
  - `figures`: Folder containing the notebooks to create the figures.
    - `beta_all_sensors.ipynb`: Notebook to obtain the Beta Power across Sensors figures.
    - `beta_power.ipynb`: Notebook to obtain the Beta Power over time figures.
    - `csp_beta_Figures.ipynb`: Notebook to obtain the CSP on Beta-Band figures.
    - `tfr_sfp_Figures.ipynb`: Notebook to obtain the TFR and SFP figures.
  - `results`: Folder containing the notebook to create the Table with Accuracy results.
    - `classifier,ipynb`: Notebook to obtain the results for the Finger Discrimination Classification table.
- `opm_thesis`: Folder containing all the files to pre-process, analyze and classify the data.
  - `classifiers`: Folder containing the files to perform binary and multi-class classification of the data.
    - `csp`: Folder containing the files to perform CSP.
      - `csp_classifiers.py`: File to classify the data using CSP and LDA.
      - `csp_functions.py`: File containig the functions to classify data and create the plots.
      - `resting_csp.py`: File containing the script to classify the data using CSP and LDA on resting epochs.
    - `classifier_functions.py`: File containing the function that use the deep-learning models to classify the data.
    - `classifiers.py`: File containing the deep-learning models.
  - `create_epochs`: Folder containing the files to filter the epochs in a specific frequency and create a unified epochs file.
    - `epochs_freq_bands.py`: File containing the script to filter the epochs and unify them in an epochs file.
    - `epochs_resting.py`: Same as above but for resting data.
  - `preprocessing`: Folder containing preprocessing files.
    - `finding_events.py`: File applying preprocessing to the FIFF files.
    - `preprocessing.py`: File containing the preprocessing steps.
    - `utils.py`: File containing the utils functions for the preprocessing.
  - `read_files`: Folder containing the files to read the cMEG file into a FIFF file.
    - `cMEG2fiff_bespoke.py`: File containing the function to turn the cMEG file into a FIFF one.
    - `utils.py`: File containing the utils functions for the cMEG2FIFF file.
   
## Abstract

Stroke significantly impairs upper limb function, directly affecting quality of life. Brain-Computer Interfaces (BCIs) processing brain signals to control external devices offer promising rehabilitation solutions. However, the spatial resolution limitations of the standard recording technique, electroencephalography (EEG), hinder the decoding of brain activity.
\

This research explores the use of magnetoencephalography (MEG) with Optically Pumped Magnetometers (OPMs), which provide higher spatial resolution, focusing on the potential for discriminating finger movements. By investigating the capabilities of OPM-MEG through frequency analysis, spatial filtering, and deep learning, this thesis aims to advance BCI technology, focusing on decoding finger movement-related neural activity.
\

Initial findings in this study hinted at potential for binary classification, particularly in the beta band (\SIrange{12.5}{30}{\hertz}). Frequency-based analysis, focused on that band, underscored the difficulty in achieving reliable discrimination via visualization. Using more complex approaches, a notable outcome was achieved with a Low-Frequency Convolutional Neural Network (LF-CNN), revealing a classification accuracy of 76\% between two specific finger movements. This result is comparable to state-of-the-art findings using EEG. Multi-class discrimination among five fingers, on the other hand, proved to be challenging since the differences in brain activity were too subtle.
\

The study shows that although OPMs provide high spatial resolution, their effectiveness in complex BCI tasks such as fine-grained finger movement discrimination remains limited. This highlights the necessity for additional research and the potential benefits of integrating OPM data with other modalities to improve classification performance.

  
 
   

