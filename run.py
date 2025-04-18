# Split the book into paragraph chunks
with open('white_for_witching.txt', 'r', encoding='utf-8') as file:
    text = file.read()
paragraphs = re.split(r'\n\s*\n', text)

# Create a feature vector for each paragraph
features = []
for paragraph in paragraphs:
    # Stylometric features
    avg_sentence_length = average_sentence_length(paragraph)
    lexical_diversity = len(set(word_tokenize(paragraph.lower()))) / len(word_tokenize(paragraph))
    
    # Part-of-speech patterns (using spaCy)
    doc = nlp(paragraph)
    pos_distribution = Counter([token.pos_ for token in doc])
    
    # Function word usage (pronouns often indicate narrator perspective)
    pronoun_freq = count_pronouns(paragraph) / len(word_tokenize(paragraph))
    
    # Combine all features into a vector
    paragraph_features = [avg_sentence_length, lexical_diversity, 
                         pos_distribution['NOUN'], pos_distribution['VERB'],
                         pos_distribution['ADJ'], pronoun_freq, ...]
    
    features.append(paragraph_features)
    
# Using Gaussian Mixture Models for soft clustering
from sklearn.mixture import GaussianMixture

# Estimate number of narrators (you could also try different values)
n_clusters = 4  # For example

gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(features)

# Get cluster assignments and probabilities
labels = gmm.predict(features)
probabilities = gmm.predict_proba(features)

# Confidence scores (max probability for each paragraph)
confidence = np.max(probabilities, axis=1)