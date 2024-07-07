# Getting Started
1. install required libraries
2. put YOLO model (`yolo_tiny_configs`) into `python/yolo` directory
3. put experiment images into `input_folder`


## using the CLI
to connect to the CLI, launch the LearnerLab, click `Start Lab`,  
click `AWS Details`, copy the CLI-creds into `~/.aws/credentials`   
run `aws s3 ls` to check if it works.

# Running the experiments
## Local
## Remote
Requires valid AWS CLI credentials!  
Go into the file `YOLO_AWS_experiment.py`.  
Edit the `main` function and let it either execute 
`run_experiment(100)`, `run_experiment(1000)` or `test_single(YOUR IMAGE PATH HERE)`  