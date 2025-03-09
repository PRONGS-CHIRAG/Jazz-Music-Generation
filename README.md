# Jazz Music Generation

## Overview
This project demonstrates how to generate jazz music using a Long Short-Term Memory (LSTM) network. The model is trained on professional jazz performances and learns to improvise solos based on input sequences.

## Features
- Uses an LSTM-based recurrent neural network to generate jazz solos.
- Trained on preprocessed musical data in the form of numerical sequences.
- Outputs MIDI sequences that can be converted into playable jazz solos.
- Implements deep learning techniques with TensorFlow and Keras.

## Dataset
The dataset consists of musical values extracted from jazz performances:
- **`X`**: A (m, 30, 90) shaped array containing input sequences, where:
  - `m` is the number of training examples.
  - `T_x = 30` represents the sequence length.
  - `90` is the number of unique possible values (one-hot encoded notes/chords).
- **`Y`**: The same as `X`, but shifted one step to the left for supervised training.
- **`n_values`**: The number of unique values in the dataset (90 total).
- **`indices_values`**: A mapping of integer indices to musical values.

## Dependencies
To run this project, install the following Python packages:
```bash
pip install tensorflow numpy music21 matplotlib
```

## How to Run
1. Clone this repository and navigate to the project directory.
2. Open and execute the Jupyter Notebook `Improvise_a_Jazz_Solo_with_an_LSTM_Network_v4.ipynb`.
3. Train the model on the dataset.
4. Generate new jazz solos using the trained model.

## Model Architecture
- **LSTM Layers**: Capture sequential dependencies in musical notes.
- **Dense Layers**: Convert LSTM outputs into probabilities over possible notes.
- **Softmax Activation**: Determines the next note probabilistically.
- **Categorical Crossentropy Loss**: Optimizes the model during training.

## Results
The trained model can generate realistic jazz improvisations by predicting note sequences in real-time. The generated MIDI files can be converted into playable formats for evaluation.

## Future Improvements
- Expand dataset with more jazz recordings.
- Fine-tune hyperparameters for improved realism.
- Implement real-time MIDI generation and playback.

## Credits
This project is inspired by deep learning applications in music generation. The implementation is based on research in sequence modeling with LSTMs.

## License
This project is open-source and available for use under the MIT License.

---
**Author:** Chirag N Vijay  



