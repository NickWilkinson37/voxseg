# Voxseg

Voxseg is a Python library for voice activity detection (VAD), for speech/non-speech audio segmentation.

## Installation

This library is still in the early stage of development. Source code is avaliable on [github](https://github.com/NickWilkinson37/voxseg).

To intall this package clone the repository from GitHub to a directory of your choice:
```bash
git clone https://github.com/NickWilkinson37/voxseg.git
```
Then install using pip from the directory where you downloaded the package:
```bash
pip install ./voxseg
```

To test the installation run:
```bash
cd voxseg
python -m unittest
```
You should see two DataFrames printed, one containing normalized features, and the other with model generated endpoints.

In future installation directly from the package manager will be supported.

## Usage

May be imported and used within python scripts/modules:
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
Alternatively may be used through a command-line interface:
```bash
# reads Kaldi-style data directory and extracts features to .h5 file in output directory
python extract_feats.py data_directory output_directory
# runs VAD and saves output to .h5 file in ouput directory
python run_cnnlstm.py model_path features_directory output_directory
```

## License
[MIT](https://choosealicense.com/licenses/mit/)