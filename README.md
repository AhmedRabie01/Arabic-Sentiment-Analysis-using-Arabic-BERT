# Arabic Sentiment Analysis using AraBERT 


Welcome to the Arabic Sentiment Analysis project powered by AraBERT! This repository contains an end-to-end Natural Language Processing (NLP) project that focuses on analyzing sentiment in Arabic text. Leveraging the power of AraBERT, a state-of-the-art language model for the Arabic language, we have built a robust sentiment analysis system capable of understanding and classifying emotions within Arabic text.

# What is Sentiment Analysis?
Sentiment analysis is a fascinating area of NLP that involves understanding and classifying the emotions expressed in a given text. It allows us to determine whether a piece of text conveys a positive, negative, or neutral sentiment, providing valuable insights into public opinion, customer feedback, and more.


# project's screenshot
![project web app](https://github.com/AhmedRabie01/Arabic-Sentiment-Analysis-using-Arabic-BERT/blob/main/photo/Screenshot_12-12-2024_21316_127.0.0.1.jpeg)
![project web app](https://github.com/AhmedRabie01/Arabic-Sentiment-Analysis-using-Arabic-BERT/blob/main/photo/Screenshot_12-12-2024_21233_127.0.0.1.jpeg)



# Features
- User-friendly web interface.
- Analyze text reviews to determine sentiment.
- Outputs sentiment classification (Positive/Negative) and confidence percentage.

#  Project's Stages

### Data Ingestion from MongoDB:
We have implemented data ingestion functionalities to fetch Arabic text data from MongoDB, making it easy to work with large datasets efficiently.

### Data Validation and Transformation: 
Prior to model training, we perform data validation and transformation to ensure the dataset is clean, balanced, and suitable for training the sentiment analysis model.

### Fine-tuned AraBERT  Model: 
We fine-tuned the ARBERT model using a large dataset of labeled Arabic text for sentiment analysis. This ensures that the model can accurately capture the nuances of sentiment in Arabic language content.

### Model Training and Evaluation:
The fine-tuned model undergoes extensive training using the prepared dataset. We evaluate the model's performance using various metrics to assess its accuracy and generalization capabilities.

### Model Pusher:
Once the model training and evaluation are complete, we implement a model pusher component that allows for easy deployment of the trained model to the web application.

### Web Application Development:
We have developed a FastAPI web service that serves as the backbone of our sentiment analysis application. The web app interacts with the sentiment analysis model and allows users to input Arabic text to obtain sentiment analysis results in real time.

# Setup and Installation
1) Clone the Repository

```bash
git clone https://github.com/AhmedRabie01/Arabic-Sentiment-Analysis-using-Arabic-BERT.git

```
2) Set Up Virtual Environment 
```bash 
conda create --name myenv -c conda-forge python=3.11
```
3) Activate your Environment 
```bash
Conda activate -name  of your Environment-
``` 
4. Install Dependencies
```bash
pip install -r requirements.txt
```

5. Start the Application
```bash
uvicorn main:app --reload
```
### Notes:

- If you need to train the model you should set up MongoDB and provide the necessary connection details in the configuration files to enable data ingestion if you need to trian the model 

### Contributions and Issues
I welcome contributions from the community! 

Let's unlock the power of Arabic BERT for sentiment analysis together! Happy analyzing! :rocket:

Disclaimer: This project is for research and educational purposes only. The provided sentiment analysis results may not be 100% accurate and should be used with caution in critical applications.
