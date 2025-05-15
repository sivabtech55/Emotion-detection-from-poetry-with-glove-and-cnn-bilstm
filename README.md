# Emotion-detection-from-poetry-with-glove-and-cnn-bilstm


This project uses Glove nad CNN BiLSTM to detect emotions in short poetic or expressive texts. The model can classify text into six basic emotions: sadness, joy, love, anger, fear, and surprise.

## Dataset
We use the [Emotion Dataset by Dair-ai](https://huggingface.co/datasets/dair-ai/emotion) hosted on Hugging Face. This dataset consists of thousands of short English texts annotated with emotional labels.

## Features
- Text preprocessing
- Tokenization and vectorization
- Model training using scikit-learn or deep learning
- Evaluation using metrics like accuracy and F1-score

## Setup Instructions

```bash
pip install -r requirements.txt
python main.py
