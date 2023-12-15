# Background

As a talent sourcing and management company, our goal is to identify skilled individuals for technology companies. However, this task is challenging due to the need for a deep understanding of roles, client needs, and what makes a candidate ideal. Additionally, the manual nature of our operations poses a significant workload. To address these challenges, we aim to automate and enhance our approach. Our objective is to create a more efficient process that saves time, identifies potential candidates fitting specific roles, and utilizes a machine learning-powered pipeline to assess and rank candidates based on their fitness for a given role.

We are currently semi-automatically sourcing a few candidates, so the sourcing part is not a concern at this time. However, we expect to first determine the best-matching candidates based on how fit they are for a given role. We generally make these searches based on keywords such as “full-stack software engineer,” “engineering manager,” or “aspiring human resources,” depending on the role we are trying to fill. These keywords might change, and you can expect that specific keywords will be provided to you.

Assuming that we were able to list and rank fitting candidates, we then employ a review procedure. As each candidate needs to be reviewed and then determined how good a fit they are through manual inspection. This procedure is done manually, and at the end of this manual review, we might choose not the first fitting candidate in the list but maybe the 7th candidate in the list. If that happens, we are interested in being able to re-rank the previous list based on this information. This supervisory signal is going to be supplied by starring the 7th candidate in the list. Starring one candidate actually sets this candidate as an ideal candidate for the given role. Then, we expect the list to be re-ranked each time a candidate is starred.

## Data Description

The data comes from our sourcing efforts. We removed any field that could directly reveal personal details and gave a unique identifier for each candidate.

- **id**: unique identifier for candidate (numeric)
- **job_title**: job title for candidate (text)
- **location**: geographical location for candidate (text)
- **connections**: number of connections candidate has, 500+ means over 500 (text)

**Output (desired target):**
- **fit**: how fit the candidate is for the role? (numeric, probability between 0-1)

**Keywords:** “Aspiring human resources” or “seeking human resources”

# Methodology

- Tokenize the job title corpus using the NLTK module in Python.
- Transform job titles into vectors using pre-trained models:
    - FastText
    - Word2Vec
    - TF-IDF
    - GloVe
    - Google Bert model
- Obtain cosine similarity between the candidate’s vector and the target job keyword.

# Summary

The final model, utilizing the "Google Bert" approach, demonstrated exceptional performance in ranking and re-ranking candidates.
