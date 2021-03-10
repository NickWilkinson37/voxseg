# Voxseg

Voxseg is a Python package for voice activity detection (VAD), for speech/non-speech audio segmentation. It provides a full VAD pipeline, including a pretrained VAD model, and it is based on work presented [here](https://arxiv.org/abs/2103.03529).

Use of this VAD may be cited as follows:
```
@inproceedings{cnnbilstm_vad,
    title = {A hybrid {CNN-BiLSTM} voice activity detector},
    author = {Wilkinson, N. and Niesler, T.},
    booktitle = {Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
    year = {2021},
    address = {Toronto, Canada},
}
```

## Installation

To install this package, clone the repository from GitHub to a directory of your choice and install using pip:
```bash
git clone https://github.com/NickWilkinson37/voxseg.git
pip install ./voxseg
```
In future, installation directly from the package manager will be supported.

To test the installation run:
```bash
cd voxseg
python -m unittest
```
The test will run the full VAD pipeline on two example audio files. This pipeline includes feature extraction, VAD and evaluation. The output should include the following*:
- A progress bar monitoring the feature extraction process, followed by a DataFrame containing normalized-features and metadata.
- A DataFrame containing model generated endpoints, indicating the starts and ends of discovered speech utterances.
- A confusion matrix of speech vs non-speech, with the following values: TPR 0.935, FPR 0.137, FNR 0.065, FPR 0.863

*The order in which these outputs appear may vary.

## Data Preparation
Before using the VAD, a number of files need to be created to specify the audio that one wishes to process. These files are the same as those used by the Kaldi toolkit. Extensive documentation on the data preparation process for Kaldi may be found [here](https://kaldi-asr.org/doc/data_prep.html). Only the files required by the Voxseg toolkit are described here.

1. `wav.scp` - this file provides the paths to the audio files one wishes to process, and assigns them a unique recording-id. It is structured as follows:
    `<recording-id> <extended-filename>`
    Each entry should appear on a new line, for example:
    ```
    rec_000 wavs/some_raw_audio.wav
    rec_001 wavs/some_more_raw_audio.wav
    rec_002 wavs/yet_more_raw_audio.wav
    ```
    Note that the `<extended-filename>` may be an absolute path or relative path, except when using Docker or Singularity, where paths relative to the mount point must be used.

2. `segments` - this file is optional and specifies segments within the audio file to be processed by the VAD (useful if one only wants to run the VAD on a subset of the full audio files). If this file is not present the full audio files will be processed. This file is structured as follows:
    `<utterance-id> <recording-id> <segment-begin> <segment-end>`
    where `<segment-begin>` and `<segment-end>` are in seconds. Each entry should appear on a new line, for example:
    ```
    rec_000_part_1 rec_000 20.5 142.6
    rec_000_part_2 rec_000 362.1 421.0
    rec_001_part_1 rec_001 45.9 89.4
    rec_001_part_2 rec_001 97.7 130.0
    rec_001_part_3 rec_001 186.9 241.8
    rec_002_full rec_002 0.0 350.0
    ```

These two files should be placed in the same directory, usually named `data`, however you may give it any name. This is the directory that is provided as input to voxseg’s feature extraction.

## Usage
The package may be used in a number of ways:
1. The full VAD can be run with a single script.
2. Smaller scripts may be called to run different parts of the pipeline separately, for example feature extraction, then VAD. Useful if one is tuning the parameters of the VAD, and would like to avoid recomputing the features for every experiment.
3. As a module within python, useful if one would like to integrate parts of the system into one's own python code.

### Full VAD pipeline
This package may be used through a basic command-line interface. To run the full VAD pipeline with default settings, navigate to the voxseg directory and call:
```bash
# data_directory is Kaldi-style data directory and output_directory is destination for segments file 
python3 voxseg/main.py data_directory output_directory
```

To explore the available flags for changing settings navigate to the voxseg directory and call:
```bash
python3 voxseg/main.py -h
```
The most commonly used flags are:
* -s sets the speech vs non-speech decision threshold (accepts float between 0 and 1, default is 0.5)
* -f: adds median filtering to smooth the output (accepts odd integer for kernal size, default is 1)
* -e: allows a reference directory to be given, against which the VAD output is scored (accepts path to Kaldi-style directory containing ground truth segments file)

### Individual scripts
To run the smaller, individual scripts, navigate to the voxseg directory and call:
```bash
# reads Kaldi-style data directory and extracts features to .h5 file in output directory
python3 voxseg/extract_feats.py data_directory output_directory
# runs VAD and saves output segments file in ouput directory
python3 voxseg/run_cnnlstm.py -m model_path features_directory output_directory
# reads Kaldi-style data directory used as VAD input, the VAD output directory and a directory 
# contining a ground truth segments file reference. 
python3 voxseg/evaluate.py vad_input_directory vad_out_directory ground_truth_directory
```

### Module within Python
To import the module an use it within custom Python scripts/modules:
```python
import voxseg
from tensorflow.keras import models

# feature extraction
data = extract_feats.prep_data('path/to/input/data') # prepares audio from Kaldi-style data directory
feats = extract_feats.extract(data) # extracts log-mel filterbank spectrogram features
normalized_feats = extract_feats.normalize(norm_feats) # normalizes the features

#model execution
model = models.load_model('path/to/model.h5') # loads a pretrained VAD model
predicted_labels = run_cnnlstm.predict_labels(model, normalized_feats) # runs the VAD model on features
utils.save(predicted_labels, 'path/for/output/labels.h5') # saves predicted labels to .h5 file
```

## License
[MIT](https://choosealicense.com/licenses/mit/)