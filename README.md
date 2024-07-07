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
1.  Create a New Virtual Environment (Recommended)
2. Install the Required Packages
```
pip install -r requirements.txt
```
3. Start the Flask Server
```
python app.py
```
4. Run client
```
python client.py input_folder http://127.0.0.1:5000/object_detection
```
## Remote
Requires valid AWS CLI credentials!  
Go into the file `YOLO_AWS_experiment.py`.  
Edit the `main` function and let it either execute 
`run_experiment(100)`, `run_experiment(1000)` or `test_single(YOUR IMAGE PATH HERE)`  