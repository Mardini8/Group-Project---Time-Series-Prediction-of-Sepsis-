Assignment X
==============================

This is the file structure that should be used for the submission of the assignments.  
Submissions that do not comply with this structure will not be accepted.  
If any student or groups of students has a specific request to submit a different structure, please contact the course responsible ahead of time.  

Submission Standards
------------
- Make sure your code can be executed again;
- Do not submit the environment folder, only the requirement list;
- Do not submit the raw data folder (we have access to the raw data and the file becomes too heavy);
- Use relative paths to read files (so that we can execute the code);
- All pre-processing steps should be in the code;
- If one pre-processing step is computationally heavy, you can _optionally_ include a processed_data folder in your submission (but make sure the steps are described in the report);

Submission Model (Project Directory)
------------

    ├── raw_data                                <- [EMPTY] The original data should be read from this folder.  
    │
    ├── processed_data                          <- [OPTIONAL] Generated processed data.  
    │
    ├── models                                  <- Trained models used in the report.  
    │   └── svm_81_2401051157                   <- Naming standard suggestion: 'type_accuracy_datetime'
    │
    ├── firstname_lastname_assingment1.ipynb    <- The report file. A single file should contain all the answers.  
    │                                              Naming standard: 'firstname_lastname_assingmentnum'.   
    │
    ├── utility_functions.py                    <- Utility functions and classes used on the report analysis.  
    │
    └── requirements.txt                        <- The requirements file for reproducing the analysis environment.  