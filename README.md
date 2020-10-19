# NNE-TCP

NNE-TCP - Neural Network Embeddings for Test Case Prioritization - is a Machine Learning tool developed in Python for Continuous Integration systems. It aims to reduce the feedback-loop between making a commit and knowing if it was successfuly merged into the mainline, by prioritizing more relevant tests.

* Finds relations between **Modified-Files** and **Test-Case** from historical data.
* Embeddings enable **Entity Representation**, grouping in the embedding space elements that are similar.
* **Lightweight** 
* **Modular** to fit different system configurations.

## Summary: 
OOP-Scheme takes raw CI Data, transforms it and cleans it. By reimplementing the Keras DataGenerator class, we can generate new batches of data on-the-fly, saving up memory.
  Then the data is used to feed a Neural Network Embedding Machine Learning Model that will be used to make predictions on unseen data, i.e. make more meaningful test orderings to catch regressions quicker.  

## Class Scheme: 

Below we have the framework depicted in UML:

[![UML](images/ClassDiagramUML.png)](http://typo3.demo.aimeos.org/)
   
* **DataCI:** Inherits from Abstract Class Data. Takes raw data input, cleans it and transforms it into (file, test) pairs. 

* **NNEmbeddings:** Implements Abstract Class Model and inherits methods from Metrics and Visualizer and corresponds to the whole Machine Learning Framework.

* **DataGenerator:** Creates meaningful and balanced batches for machine learning model input.

* **Metrics:** Utility Class for Classification and Regression Metrics.

* **Visualizer:** Utility Class for Plotting several graphs.

## How it Works
* **Raw data is loaded and transformed:** missing values are removed, data types converted, file name strings are split.
* **Data Cleaning:** Some files present in the data are deprecated or unused. Thus we only want to keep relevant files. To do that, only files that have been modified recently are kept, otherwise are removed. Drops tests/files from data that cause less than threshold transitions/modifications, i.e. very stable tests/files.
* **ML Input:** Create unified input data for ML algorithm, where columns are (revision, mod_files (list)  and test names (list). Then we want to create pairs of the form (file, test). For each revision, the lists above are exploded to make pair-wise combinations between elements. Now we need to create a training set.
* **DataGenerator:** Generates data for Keras Model. Receives pairs of (files, tests) and generates random combinations of (file, test, label). Where the label is either 1 if the pair exists in the data or 0 otherwise. The class balance is given by the negative_ratio parameter, if parameter is 1, class balance is 50% each.
* **Keras Model:** Neural Network Embeddings depicted below.
* **Evaluation:** Once the model is trained. When new revisions arrive, it is able to predict which tests will more likely be linked to the modified files and create a more meaningful test ordering to detect faults earlier. The evaluation is done by the APFD (Average Percentage of Faults Detections) metric.


[![Keras Model](images/model.png)](model)

