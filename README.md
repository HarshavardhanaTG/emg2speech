## EMG-to-speech decoding

**Paper:** *Non-invasive electromyographic speech neuroprosthesis: a geometric perspective*  
**Authors:** Harshavardhana Gowda and Lee M. Miller.  
**Link:** [https://arxiv.org/pdf/2502.05762](https://arxiv.org/pdf/2502.05762)  
**Citation:**
```
@inproceedings{
gowda2025noninvasive,
title={Non-invasive electromyographic speech neuroprosthesis: a geometric perspective},
author={Harshavardhana T Gowda and Lee M. Miller},
booktitle={NeurIPS 2025 Workshop on Learning from Time Series for Health},
year={2025},
url={https://openreview.net/forum?id=hHEMyFejP8}
}
```

This repository contains code for decoding speech using surface electromyography (sEMG) signals.  
The system captures muscle activity from the face and neck and maps it to phonemes.  
The goal is to create non-invasive neural speech interfaces. This paper contains only large-vocab results.


## 📁 Repository Structure

    ```
    emg2speech/
    ├── basicOperations/ # SPD matrix operations on the manifold.
    ├── rnn/ # GRU recurrent neural networs in Euclidean space and manifold space.
    ├── emg2qwerty/ # Code to replicate results for emg2qwerty using our method.

    ├── DATA/ Download the data and place it in this repo.

    ├── requirements.txt # Python dependencies.
    ├── README.md # Project description and instructions.
    └── .gitignore # Excludes DATA/ and other ignored files.

    ├── largeVocabTrain.ipynb # Notebook to train large-vocab corpora.
    ├── largeVocabTest.ipynb # Notebook to test the results for large-vocab corpora using a pretrained checkpoint.
    ├── largeVocabTestWithLM.ipynb # Notebook to test the results for large-vocab corpora using a pretrained checkpoint with a phone-level language model.
    ├── largeVocabTrainWithIcefall.ipynb # Notebook to train large-vocab corpora with spaces removed.
    ├── largeVocabTestWithIcefall.ipynb # Notebook to test the results for large-vocab corpora using a pretrained checkpoint using icefall WFST model.
    ├── largeVocabDataVisualization.ipynb # Notebook to visualize large-vocab corpora.
    ├── largeVocabTrainSpectrogram.ipynb # Baseline spectrogram with vanilla GRU.

    ├── smallVocabEuclidean.ipynb # Notebook to train and test small-vocab corpora using Euclidean RNN.
    ├── smallVocabManifold.ipynb # Notebook to train and test small-vocab corpora using manifold RNN.
    ├── natoWords.ipynb # Notebook to train a model for NATO words.
    ├── checkGrandfather.ipynb # Notebook to test articulation from grandfather passage using trained checkpoint from natoWords.ipynb. 
    ├── checkrainbow.ipynb # Notebook to test articulation from rainbow passage using trained checkpoint from natoWords.ipynb. 
    ```

## The `DATA/` folder contains EMG data and labels and is **not included in this repository** due to size .

To obtain the data:

1. Download the data from:  https://osf.io/bgh7t/ (under Files/Box).
   (https://osf.io/bgh7t/files/box - the OSF site shows them as 0B since they are hosted
   externally. You can still click on individual file and download them. 0B is misleading.) 
3. Unzip it into the root project directory so it looks like this:

    ```
    emg2speech/
    ├── DATA/
    │   ├── ckptsLargeVocab/ # Trained check points for data large-vocab (Also, all LMs + WFST graphs).
    │   ├── ckptsNatoWords/ # Trained check points for data NATO-words.
    │   ├── ckptsSmallVocab/ # Trained ckeck points for data small-vocab
    └── ├── emg2qwerty/ # Trained check points for emg2qwerty.
    ``` ├── dataLargeVocab.pkl # large-vocab EMG data.
        ├── labelsLargeVocab.pkl # Labels for large-vocab.
        ├── dataSmallVocab.npy # small-vocab EMG data.
        ├── labelsSmallVocab.npy # Labels for small-vocab.
        ├── Audio # Synthesized personalized audio file samples.
        ├── Subject 1 # NATO words, subject 1
        ├── Subject 2 # NATO words, subject 2
        ├── Subject 3 # NATO words, subject 3
        ├── Subject 4 # NATO words, subject 4
        
