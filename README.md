# Voxseg

Voxseg is a Python library for voice activity detection (VAD), for speech/non-speech audio segmentation.

## Installation

This library is still in the early stage of development. For now source code is avaliable on [github](https://github.com/NickWilkinson37/voxseg.git).

```bash
git clone https://github.com/NickWilkinson37/voxseg.git
```

In future installation from the package manager will be supported.

## Usage

May be imported and used within python scripts/modules:
```python
from voxseg import extract_feats

data = extract_feats.prep_data(data_directory) # prepares audio from Kaldi-style data directory
features = extract_feats.pluralize(data) # extracts log-mel filterbank spectrogram features
extract_feats.save(features, output_directory) # saves features to .h5 file in output directory
```
Alternatively may be used through a command-line interface:
```bash
# reads Kaldi-style data directory and extracts features to .h5 file in output directory
python extract_feats.py data_directory output_directory
```

## License
[MIT](https://choosealicense.com/licenses/mit/)