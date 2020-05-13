
Deep 1D-Convnet for accurate Parkinson disease detection and severity prediction from gait
-

This is the official code release for the paper : 
El Maachi, I., Bilodeau, G.-A., Bouachir, W., Deep 1D-Convnet for accurate Parkinson disease detection and severity prediction from gait, Expert Systems With Applications, Volume 143, 2020

You can access it via arxiv: https://arxiv.org/abs/1910.11509 

For any questions or queries, please contact Imanne El Maachi: imanne.elmaachi@gmail.com 

Prerequisites
-
- Python 3.7 
- CPU or  NVIDIA GPU + CUDA CuDNN ( the algorithm was developed with Cuda and CudNN )
- Install the preqrequisistes libraries: pip install -r requirements.txt

Dataset
- 
The dataset  used in the paper is from Physionet, it can be downloaded from: 
https://physionet.org/content/gaitpdb/1.0.0/ . A sample of the dataset is available in subfolder data. 

Getting Started 
-
The entry point is train.py file, it has 3 arguments: 
  - input_data: Input folder containing the dataset. 
  - exp_name: type of experiment to run: 
      - 'train_classifier': Run the cross-validation experiment for Parkinson detection. 
      - 'train_severity': Run the cross-validation experiment for Parkinson severity prediction using the UPDRS scale.
      - 'ablation': Ablation study for the different gait signals for Parkinson detection. 
  
  Once runned, the algorithm will generate the following output files:
 
 ------------

    ├── output (dir)
        ├── exp_name_Day_Month (when the  program was launched) 
            ├── hour_minutes (when the program was launched)  
                ├──  weights.hdf5 : weights of the model
                ├──  res_seg.csv: results of accuracy, sensitivity and specificity by segments.
                ├──  res_pat.csv: results of accuracy, sensitivity and specificity by patients. 
                ├──  training_i.csv: training loss and accuracy for each fold i . For 10-fold cross-validation, 10 files. 
                ├──  model.json: Model architecture (keras)
                ├──  gt.csv: and pred.csv  ( for severity prediction) ground truth level and prediction level for each patient.
                ├──  confusion_matrix.csv : Confusion matrix for severity prediction
                
                

