# Next-Word-Prediction-System

**Next Word Prediction with LSTM**

This project demonstrates a **Next Word Prediction** model using an **LSTM (Long Short-Term Memory)** neural network trained on *Hamlet by William Shakespeare*. The model predicts the most probable next word in a sequence, and is deployed via a **Streamlit web application**.

---

## Features  

- **Deep Learning Model**: Built with TensorFlow/Keras LSTM.  
- **Dataset**: Text corpus from *Hamlet* (`hamlet.txt`).  
- **Tokenizer**: Pre-trained tokenizer stored in `tokenizer.pickle`.  
- **Trained Model**: Pre-trained weights in `next_word_lstm.h5`.  
- **Interactive UI**: Streamlit app (`app.py`) lets users enter text and predict the next word in real time.  
- **Notebook Experiments**: Exploratory training and evaluation are in `experiments.ipynb`.  

---
## Project Structure  

├── app.py # Streamlit app for next word prediction
├── experiments.ipynb # Jupyter Notebook for training & analysis
├── hamlet.txt # Training corpus (Hamlet by Shakespeare)
├── next_word_lstm.h5 # Trained LSTM model
├── tokenizer.pickle # Tokenizer used for preprocessing
├── requirements.txt # List of dependencies
└── README.md # Project documentation


---
##  Installation  

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   streamlit run app.py

##  Model Details  

- **Architecture**: LSTM-based language model  
- **Training Data**: Tokenized sequences from `hamlet.txt`  
- **Sequence Length**: Derived from model input shape at runtime  
- **Output**: Predicts the most likely next word using softmax probabilities  

---

## Experiments  

The `experiments.ipynb` notebook contains:  
- Data preprocessing (tokenization, sequence padding)  
- Model training with early stopping  
- Accuracy and loss visualization  
- Hyperparameter tuning  

---

## Future Improvements  

- Extend dataset with larger corpora for better generalization  
- Implement **beam search** for multi-word predictions  
- Add **API endpoints** for external integration  
- Support for **top-k** or **temperature-based sampling**  

