import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer, util


#============================================
talents_df=pd.read_csv('C:/Users/M-ODE/Desktop/Apziva/projects/3rd Project/potential_talents/data/potential_talents_CSV.csv')
#==============================================
#merging location with job title
def merge_text(row):
    return f"{row['job_title']} ({row['location']})"

talents_df['job_title'] = talents_df.apply(merge_text, axis=1)

# After applying the function, you can drop the 'location' column if it's no longer needed.
talents_df = talents_df.drop('location', axis=1)

#==============================================
# Load a pre-trained SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

keywords = ["Aspiring human resources", "seeking human resources"]

keyword_similarities = []

# Tokenize job titles and obtain their SBERT embeddings
job_title_embeddings = model.encode(talents_df['job_title'].tolist(), convert_to_tensor=True)

for keyword in keywords:
    # Tokenize the current keyword and obtain its SBERT embedding
    keyword_embedding = model.encode(keyword, convert_to_tensor=True)
    
    # Calculate cosine similarities between the current keyword and job title embeddings
    similarities = util.pytorch_cos_sim(keyword_embedding, job_title_embeddings)
    
    # Convert the PyTorch tensor to a NumPy array before appending
    keyword_similarities.append(similarities.cpu().numpy())

# Combine or average the similarity scores based on all keywords for each job title
average_similarities = np.mean(keyword_similarities, axis=0)

# Reshape the average_similarities array to (104,)
average_similarities = average_similarities[0]

talents_df['fit'] = average_similarities.tolist()

# Sort 
ranked_data = talents_df.sort_values(by='fit', ascending=False)

print(ranked_data.head())
#====================================================


star_index = int(input("Enter the index of the starred job title: "))
# Check if the input index is valid
if star_index < 0 or star_index >= len(talents_df):
    print("Invalid index. Please enter a valid index.")
    exit()
starred_candidate = talents_df['job_title'][star_index]
#================================================

star = starred_candidate           #example; id= 32

keyword_similarities = []

# Tokenize job titles and obtain their SBERT embeddings
job_title_embeddings = model.encode(talents_df['job_title'].tolist(), convert_to_tensor=True)

for keyword in keywords:
    # Tokenize the current keyword and obtain its SBERT embedding
    keyword_embedding = model.encode(keyword, convert_to_tensor=True)
    # Tokenize the "star" snippet and obtain its BERT embeddings
    star_embedding = model.encode(star, convert_to_tensor=True)
    mean_embedding = (keyword_embedding + star_embedding) / 2
    
    # Calculate cosine similarities between the current keyword and job title embeddings
    similarities = util.pytorch_cos_sim(mean_embedding, job_title_embeddings)
    
    # Convert the PyTorch tensor to a NumPy array before appending
    keyword_similarities.append(similarities.cpu().numpy())

# Combine or average the similarity scores based on all keywords for each job title
average_similarities = np.mean(keyword_similarities, axis=0)

# Reshape the average_similarities array to (104,)
average_similarities = average_similarities[0]

talents_df['fit'] = average_similarities.tolist()

ranked_data = talents_df.sort_values(by='fit', ascending=False)

print(ranked_data.head())
