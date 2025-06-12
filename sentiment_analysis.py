from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 

# assigns by if statement checks through pyTorch lib function if a CUDA-enabled GPU is available on the system and 
# assigns the appropriate computation device (GPU or CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#loads necessary model and tokenizer for finBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

labels = ["positive", "negative", "neutral"] #sentiments labels
 
# Function to estimate sentiment from a list of news strings
def estimate_sentiment(news_list):
    if news_list:  # Check if the input list is not empty

        tokens = tokenizer(news_list, return_tensors="pt", padding=True).to(device)
        logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]

        result = torch.nn.functional.softmax(torch.sum(logits, dim=0), dim=-1) #softmax gets probabilities 

        # Extract the most probable sentiment and its probability
        probability = result[torch.argmax(result)].item()  # Convert tensor to float
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]

if __name__ == "__main__":
    news_list = [
        "Markets responded negatively to the news!",
        "Traders were displeased!",
    ]
    probability, sentiment = estimate_sentiment(news_list)
    print(f"Sentiment Probability: {probability}, Sentiment: {sentiment}")  
    print(f"CUDA Available: {torch.cuda.is_available()}") #shows if cuda enabled GPU is available