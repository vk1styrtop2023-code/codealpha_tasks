

# **Emotion Recognition from Speech 🎤💬**


## **Overview**
This project aims to recognize emotions from speech using a Long Short Term Memory (LSTM) model. It was developed during my internship as a Machine Learning Engineer at CodeAlpha.



## **Features**
- Predicts emotions such as **Fear**, **Sadness**, and more from speech audio.
- User-friendly web interface for easy interaction.
- Built using advanced machine learning techniques for accurate predictions.

## **Project Lifecycle**
1. **Data Collection** 📊
   - Gathered diverse audio samples for training the model.


2. **Data Preprocessing** 🧹
   - Cleaned and prepared the audio data for analysis.



3. **Data Augmentation** 🔄
   - Enhanced the dataset by creating variations of existing audio samples.
  


4. **Feature Extraction** 🔍
   - Identified and extracted key features from the audio files.
  

5. **Model Training** 🏋️‍♂️
   - Trained the LSTM model on the processed data.



6. **Model Evaluation** 📈
   - Assessed the model's performance and improved accuracy.



7. **Save the Model** 💾
   - Saved the trained model for future predictions.



9. **Model Deployment** 🚀
   - Deployed the model as a web application for user access.



## **Getting Started**
To run this project locally, follow these steps:

### **Prerequisites**
- Python 3.x
- Required libraries (listed in `requirements.txt`)

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/emotion-recognition-speech.git
   cd emotion-recognition-speech
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the web app:
   ```bash
   python app.py
   ```

## **📁 Repository Structure**
```
emotion-recognition-speech/
│
├── app/
│   └── app.py                # Main application file
├── docs/
│   ├── codealpha-emotion-recognition-from-speech.pdf
    ├── Emotion-Recognition-From-Speech.pdf             # Document for steps taken to make this project
├── jupyter notebook/
│   └──  codealpha-emotion-recognition-from-speech.ipynbb       # To Do Experimentation with data
├── LSTM model/
│   └── lstm_model.h5              # LSTM Model
├── test audio files/
│   ├── OAF_base_sad.wav
    ├── YAF_back_fear.wav             # Audio used for model prediction
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
└── ...
```


## **Acknowledgments**
Thanks to #CodeAlpha for the opportunity to work on this exciting project!


