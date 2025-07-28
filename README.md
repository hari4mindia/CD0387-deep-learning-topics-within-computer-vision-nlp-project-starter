# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio.  
Download the starter files.  
Download/Make the dataset available.  

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.  
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
The model used is a pretrained ResNet18, chosen for its balance between performance and training efficiency.  
The hyperparameter tuning job explored:
- `learning_rate`: [0.001, 0.01]
- `batch_size`: [32, 64]

The tuning job was configured to run multiple training jobs in parallel and select the best model based on validation accuracy.

### Screenshot of Tuning Jobs
!Tuning Job 1  
!Tuning Job 2

## Debugging and Profiling
SageMaker Debugger and Profiler were used to monitor training performance and resource utilization.  
Rules such as `vanishing_gradient`, `overtraining`, and `loss_not_decreasing` were applied to catch training issues early.

### Results
- The profiler showed consistent GPU utilization and no major bottlenecks.
- Debugger flagged a few early epochs with unstable gradients, which were resolved by adjusting the learning rate.

📎 Profiler report is included in the submission as `ProfilerReport/`.

## Model Deployment
The best model from the tuning job was deployed to a SageMaker endpoint using `PyTorchModel`.

### How to Query the Endpoint
```python
from PIL import Image
import io
import numpy as np

with open("dogImages/test/004.Akita/Akita_00244.jpg", "rb") as f:
    image = f.read()
    response = predictor.predict(image, initial_args={"ContentType": "image/jpeg"})
    prediction = np.argmax(response) + 1
    print(f"Predicted class: {prediction}")
