# Corsound_assignment

## Data
The data used for the assignment is available in https://bil.eecs.yorku.ca/datasets/.
The dataset contains directories in the following structure:

<img width="167" alt="Screen Shot 2023-04-02 at 13 56 41" src="https://user-images.githubusercontent.com/53824160/229355847-ad105612-f155-46a1-8244-755b95804675.png">

  
## Requirements 
Required packages:

1. transformers
2. datasets
3. torchaudio
4. scikit-learn
5. matplotlib
6. seaborn
7. pyeer

Note: There is a docker file available.

## Instructions

1. Clone the repository via:
<pre>
git clone https://github.com/shiritdvir/Corsound_assignment.git
</pre>
2. Download the data from https://bil.eecs.yorku.ca/datasets/ & change the path in the config
3. Install requirements file
<pre>
cd /path/to/code
pip install -r requirements.txt
</pre>
4. Run main.py
<pre>
python main.py
</pre>

To run via the docker open the terminal and run the following commands:
<pre>
cd /path/to/code
docker build -t your_image_name .
docker run -it --rm your_image_name
</pre>
Note: Change parameters in the config if needed.

## Results
The results are available in the [notbook](https://github.com/shiritdvir/Corsound_assignment/blob/main/Corsound_assignement.ipynb).
