This is my own implementation of a Transformer model for machine translation using PyTorch. 
It includes data preprocessing, model architecture, training, and some evaluation components, as well as text translation functionality.

It uses the Tatoeba dataset for English-French translation tasks.

The data is managed and preprocessed using Data_handler.py,
The model architecture is defined in Transformer.py,
and using Attention.py for attention mechanisms, Encoder.py and Decoder.py for the respective parts of the Transformer.
Training is handled in Training.py,

The jupyter notebook test_model.ipynb is provided for testing and evaluating the trained model.
It is elementary but allows to train/test the model easily.