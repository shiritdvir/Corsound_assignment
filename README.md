# Real speech classifier assignment

## Objective:
In this home test, the goal is to fit a classifier to distinguish real speech from fake.

## Data:
The data used for the assignment is available in https://bil.eecs.yorku.ca/datasets/.
The dataset contains directories in the following structure:

<img width="167" alt="Screen Shot 2023-04-02 at 13 56 41" src="https://user-images.githubusercontent.com/53824160/229355847-ad105612-f155-46a1-8244-755b95804675.png">

  
## Requirements: 
Required packages:

1. transformers
2. datasets
3. torchaudio
4. scikit-learn
5. matplotlib
6. seaborn
7. pyeer

Note: There is a docker file available.

## Instructions:

1. Clone the repository:
<pre>
git clone https://github.com/shiritdvir/Corsound_assignment.git
</pre>
2. Run the following:
```
cd Corsound_assignment
curl -O https://www.eecs.yorku.ca/~bil/Datasets/for-2sec.tar.gz
mkdir -p ./data && tar -xzf ./for-2sec.tar.gz -C ./data/
```
### Run locally
<pre>
pip install -r requirements.txt
python main.py
</pre>

### Run via docker
> When running on a GPU-supported machine, change the Dockerfile line `FROM python:3.9-slim` to `FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel`.
<pre>
docker build -t corsound_assignmnet .
docker run -it --rm -v PATH_TO_DATA:/app/data/for-2seconds corsound_assignmnet
</pre>
Replace `PATH_TO_DATA` with the absolute local path to for-2seconds dataset.

Note: Modify the parameters in the configuration as needed (such as epochs, learning rate, etc.).

## Results:
The results are available in the [notbook](https://github.com/shiritdvir/Corsound_assignment/blob/main/Corsound_assignement.ipynb).
