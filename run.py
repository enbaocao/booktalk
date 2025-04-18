import re
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import spacy
from collections import Counter
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import pandas as pd
import seaborn as sns
import warnings

# Suppress specific sklearn warning about n_iter vs max_iter
warnings.filterwarnings("ignore", message="'n_iter' was renamed to 'max_iter'", category=FutureWarning)
import warnings

# Suppress specific sklearn warning about n_iter vs max_iter
warnings.filterwarnings("ignore", message="'n_iter' was renamed to 'max_iter'", category=FutureWarning)

# Check and download required resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt', quiet=True)

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

def load_and_split_text(file_path):
    """Load text file and split into sentences instead of paragraphs"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        # Split text into sentences using NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
        # Remove very short sentences (likely formatting artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        print(f"Total sentences found: {len(sentences)}")
        # Preview first few sentences
        if sentences:
            print("First few sentences:")
            for i, s in enumerate(sentences[:3]):
                print(f"  {i+1}: {s[:100]}{'...' if len(s) > 100 else ''}")
        return sentences
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def average_sentence_length(text):
    """Calculate sentence length in words
    When analyzing at sentence level, this is just the word count"""
    # For sentence-level analysis, each text is already a single sentence
    words = word_tokenize(text)
    return len(words)

def lexical_diversity(text):
    """Calculate lexical diversity (type-token ratio)"""
    words = word_tokenize(text.lower())
    if not words:
        return 0
    return len(set(words)) / len(words)

def count_pronouns(text):
    """Count personal pronouns"""
    pronouns = ['i', 'me', 'my', 'mine', 'myself', 
                'you', 'your', 'yours', 'yourself', 
                'he', 'him', 'his', 'himself', 
                'she', 'her', 'hers', 'herself',
                'it', 'its', 'itself',
                'we', 'us', 'our', 'ours', 'ourselves',
                'they', 'them', 'their', 'theirs', 'themselves']
    words = word_tokenize(text.lower())
    return sum(1 for word in words if word in pronouns)

def first_person_ratio(text):
    """Calculate ratio of first-person pronouns to all pronouns"""
    first_person = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
    words = word_tokenize(text.lower())
    first_person_count = sum(1 for word in words if word in first_person)
    pronoun_count = count_pronouns(text)
    if pronoun_count == 0:
        return 0
    return first_person_count / pronoun_count

def extract_features(sentences):
    """Extract stylometric features from each sentence"""
    features = []
    feature_names = [
        'avg_sentence_length',
        'lexical_diversity',
        'word_count',
        'pronoun_ratio',
        'first_person_ratio',
        'noun_percent',
        'verb_percent',
        'adj_percent',
        'adv_percent',
        'det_percent',
        'avg_word_length',
        'punct_density',
        'sentence_count',
        'question_ratio',
        'exclamation_ratio',
        'semicolon_ratio'
    ]
    
    for sentence in sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        # Basic metrics
        avg_sent_len = average_sentence_length(sentence)  # For sentences, this is the same as word count
        lex_div = lexical_diversity(sentence)
        words = word_tokenize(sentence.lower())
        word_count = len(words)
        
        # Pronoun usage
        pronoun_count = count_pronouns(sentence)
        pronoun_ratio = pronoun_count / word_count if word_count > 0 else 0
        first_person_pct = first_person_ratio(sentence)
        
        # POS distribution using spaCy
        doc = nlp(sentence)
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        
        # Calculate POS percentages
        noun_pct = pos_counts['NOUN'] / total_tokens if total_tokens > 0 else 0
        verb_pct = pos_counts['VERB'] / total_tokens if total_tokens > 0 else 0
        adj_pct = pos_counts['ADJ'] / total_tokens if total_tokens > 0 else 0
        adv_pct = pos_counts['ADV'] / total_tokens if total_tokens > 0 else 0
        det_pct = pos_counts['DET'] / total_tokens if total_tokens > 0 else 0
        
        # Average word length
        avg_word_len = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Punctuation density
        punct_count = sum(1 for token in doc if token.is_punct)
        punct_density = punct_count / word_count if word_count > 0 else 0
        
        # For single sentences, this is always 1, but keep for consistency
        sent_count = 1
        
        # Special punctuation ratios
        question_marks = sentence.count('?')
        exclamation_marks = sentence.count('!')
        semicolons = sentence.count(';')
        
        total_punct = punct_count if punct_count > 0 else 1  # Avoid division by zero
        
        question_ratio = question_marks / total_punct
        exclamation_ratio = exclamation_marks / total_punct
        semicolon_ratio = semicolons / total_punct
        
        # Combine all features
        feature_vector = [
            avg_sent_len,
            lex_div,
            word_count,
            pronoun_ratio,
            first_person_pct,
            noun_pct,
            verb_pct,
            adj_pct,
            adv_pct,
            det_pct,
            avg_word_len,
            punct_density,
            sent_count,
            question_ratio,
            exclamation_ratio,
            semicolon_ratio
        ]
        
        features.append(feature_vector)
    
    return np.array(features), feature_names

def cluster_paragraphs(features, n_clusters=4):
    """Cluster sentences using Gaussian Mixture Model"""
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply GMM
    gmm = GaussianMixture(n_components=n_clusters, 
                          covariance_type='full', 
                          random_state=42,
                          n_init=10)
    gmm.fit(scaled_features)
    
    # Get cluster assignments and probabilities
    labels = gmm.predict(scaled_features)
    probabilities = gmm.predict_proba(scaled_features)
    
    # Calculate confidence (max probability)
    confidence = np.max(probabilities, axis=1)
    
    return labels, probabilities, confidence, scaled_features, gmm

def find_ambiguous_sections(sentences, labels, confidence, probabilities, threshold=0.6):
    """Identify sentences with ambiguous narrator assignment"""
    ambiguous_sentences = []
    
    for i, conf in enumerate(confidence):
        if conf < threshold:
            ambiguous_sentences.append({
                'index': i,
                'text': sentences[i],
                'confidence': float(conf),
                'assigned_cluster': int(labels[i]),
                'probabilities': probabilities[i].tolist()
            })
    
    return ambiguous_sentences

def analyze_narrator_flow(labels, confidence):
    """Analyze narrator transitions throughout the text"""
    transitions = []
    
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            transitions.append({
                'from': int(labels[i-1]),
                'to': int(labels[i]),
                'position': i,
                'confidence_before': float(confidence[i-1]),
                'confidence_after': float(confidence[i])
            })
    
    return transitions

def visualize_clusters(scaled_features, labels, confidence, output_dir='output'):
    """Visualize clusters using t-SNE"""
    # Reduce dimensions for visualization
    # Adjust perplexity to be less than number of samples
    perplexity_value = min(30, len(scaled_features) - 1)
    print(f"Using perplexity value of {perplexity_value} for t-SNE (based on {len(scaled_features)} samples)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, max_iter=1000)
    reduced_features = tsne.fit_transform(scaled_features)
    
    # Create dataframe for seaborn
    df = pd.DataFrame({
        'x': reduced_features[:, 0],
        'y': reduced_features[:, 1],
        'cluster': labels,
        'confidence': confidence
    })
    
    # Plot with confidence as opacity
    plt.figure(figsize=(12, 10))
    
    # Create a unique color for each cluster
    palette = sns.color_palette('viridis', len(np.unique(labels)))
    
    # Plot each point
    scatter = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='cluster',
        palette=palette,
        size='confidence',
        sizes=(50, 200),
        alpha=0.7
    )
    
    plt.title('Narrator Clusters Visualization', fontsize=16)
    plt.xlabel('t-SNE dimension 1', fontsize=12)
    plt.ylabel('t-SNE dimension 2', fontsize=12)
    plt.legend(title='Narrator Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'narrator_clusters.png'), dpi=300)
    plt.close()
    
    # Create confidence timeline
    plt.figure(figsize=(15, 6))
    
    # Create scatter plot with time on x-axis, narrator on y-axis, and confidence as alpha
    for i in range(len(labels)):
        plt.scatter(i, labels[i], c=palette[labels[i]], alpha=confidence[i], s=50)
    
    plt.title('Narrator Transitions Throughout the Text', fontsize=16)
    plt.xlabel('Sentence Index', fontsize=12)
    plt.ylabel('Assigned Narrator', fontsize=12)
    plt.yticks(np.unique(labels))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'narrator_timeline.png'), dpi=300)
    plt.close()
    
    # Create confidence histogram
    plt.figure(figsize=(10, 6))
    for i in np.unique(labels):
        cluster_conf = confidence[labels == i]
        plt.hist(cluster_conf, alpha=0.7, label=f'Narrator {i}', bins=20)
    
    plt.title('Confidence Distribution by Narrator', fontsize=16)
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300)
    plt.close()

def analyze_feature_importance(model, feature_names, output_dir='output'):
    """Analyze feature importance for each cluster"""
    # Extract means for each cluster
    means = model.means_
    
            # Create a heatmap of feature importance by cluster
    plt.figure(figsize=(14, 8))
    sns.heatmap(means, annot=False, cmap='coolwarm', 
                xticklabels=feature_names, 
                yticklabels=[f'Narrator {i}' for i in range(means.shape[0])])
    plt.title('Feature Importance by Narrator Cluster', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()

def save_results(sentences, labels, confidence, probabilities, ambiguous_sections, 
                transitions, output_dir='output'):
    """Save analysis results to files"""
    # Save sentence assignments
    results = []
    for i, sentence in enumerate(sentences):
        results.append({
            'sentence_index': i,
            'text': sentence[:100] + '...' if len(sentence) > 100 else sentence,
            'assigned_narrator': int(labels[i]),
            'confidence': float(confidence[i]),
            'probabilities': probabilities[i].tolist()
        })
    
    # Create DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'narrator_analysis_results.csv'), index=False)
    
    # Save ambiguous sections
    if ambiguous_sections:
        df_ambiguous = pd.DataFrame(ambiguous_sections)
        df_ambiguous.to_csv(os.path.join(output_dir, 'ambiguous_narrators.csv'), index=False)
    
    # Save transitions
    if transitions:
        df_transitions = pd.DataFrame(transitions)
        df_transitions.to_csv(os.path.join(output_dir, 'narrator_transitions.csv'), index=False)

def generate_summary_report(sentences, labels, confidence, ambiguous_sections, 
                           transitions, output_dir='output'):
    """Generate a summary report of findings"""
    unique_narrators = len(np.unique(labels))
    total_sentences = len(sentences)
    ambiguous_count = len(ambiguous_sections)
    transition_count = len(transitions)
    
    avg_confidence = np.mean(confidence)
    narrator_distribution = pd.Series(labels).value_counts().sort_index()
    
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("=== NARRATOR ANALYSIS SUMMARY ===\n\n")
        f.write(f"Total sentences analyzed: {total_sentences}\n")
        f.write(f"Number of identified narrators: {unique_narrators}\n")
        f.write(f"Average narrator confidence: {avg_confidence:.2f}\n\n")
        
        f.write("Narrator distribution:\n")
        for narrator, count in narrator_distribution.items():
            percentage = (count / total_sentences) * 100
            f.write(f"  Narrator {narrator}: {count} sentences ({percentage:.1f}%)\n")
        
        f.write(f"\nAmbiguous narrator sections: {ambiguous_count} ({(ambiguous_count/total_sentences)*100:.1f}%)\n")
        f.write(f"Narrator transitions: {transition_count}\n\n")
        
        f.write("=== TOP AMBIGUOUS SECTIONS ===\n\n")
        # Sort by confidence (ascending)
        top_ambiguous = sorted(ambiguous_sections, key=lambda x: x['confidence'])[:5]
        for i, section in enumerate(top_ambiguous):
            f.write(f"Ambiguous Section {i+1} (Confidence: {section['confidence']:.2f}):\n")
            f.write(f"  Paragraph Index: {section['index']}\n")
            f.write(f"  Assigned Narrator: {section['assigned_cluster']}\n")
            f.write(f"  Text: {section['text'][:200]}...\n\n")
        
        f.write("=== KEY NARRATOR TRANSITIONS ===\n\n")
        # Get transitions with low confidence
        key_transitions = sorted(transitions, key=lambda x: x['confidence_after'])[:5]
        for i, trans in enumerate(key_transitions):
            f.write(f"Transition {i+1} (Confidence After: {trans['confidence_after']:.2f}):\n")
            f.write(f"  Position: Paragraph {trans['position']}\n")
            f.write(f"  From Narrator {trans['from']} to Narrator {trans['to']}\n")
            f.write(f"  Confidence Before: {trans['confidence_before']:.2f}\n\n")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = ['spacy', 'nltk', 'sklearn', 'pandas', 'matplotlib', 'seaborn']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("Please install all required dependencies:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def main():
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
        
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Analyze narrators in a text using unsupervised learning.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--clusters', '-c', type=int, default=4, help='Number of narrator clusters to find')
    parser.add_argument('--threshold', '-t', type=float, default=0.6, 
                        help='Confidence threshold for ambiguous narrator detection (default lowered to 0.6 for sentence analysis)')
    parser.add_argument('--output', '-o', type=str, default='output', 
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    print("Loading and processing text...")
    sentences = load_and_split_text(args.input)
    
    if not sentences:
        print("No sentences found. Check the file path and format.")
        return
    
    print(f"Extracted {len(sentences)} sentences.")
    
    print("Extracting stylometric features...")
    features, feature_names = extract_features(sentences)
    
    print("Clustering sentences...")
    labels, probabilities, confidence, scaled_features, model = cluster_paragraphs(features, args.clusters)
    
    print("Finding ambiguous narrator sections...")
    ambiguous_sections = find_ambiguous_sections(sentences, labels, confidence, 
                                                probabilities, args.threshold)
    print(f"Found {len(ambiguous_sections)} ambiguous sections.")
    
    print("Analyzing narrator transitions...")
    transitions = analyze_narrator_flow(labels, confidence)
    print(f"Found {len(transitions)} narrator transitions.")
    
    print("Visualizing results...")
    visualize_clusters(scaled_features, labels, confidence, args.output)
    
    print("Analyzing feature importance...")
    analyze_feature_importance(model, feature_names, args.output)
    
    print("Generating summary report...")
    generate_summary_report(sentences, labels, confidence, ambiguous_sections, 
                           transitions, args.output)
    
    print("Saving detailed results...")
    save_results(sentences, labels, confidence, probabilities, 
                ambiguous_sections, transitions, args.output)
    
    print("\nAnalysis complete! Check the output files in the", args.output, "directory:")
    print("- narrator_clusters.png: Visual representation of narrator clusters")
    print("- narrator_timeline.png: Narrator transitions throughout the text")
    print("- confidence_distribution.png: Distribution of confidence scores by narrator")
    print("- feature_importance.png: Importance of different features for each narrator")
    print("- summary_report.txt: Summary of key findings")
    print("- narrator_analysis_results.csv: Complete analysis results")
    print("- ambiguous_narrators.csv: Sentences with ambiguous narrator assignment")
    print("- narrator_transitions.csv: Transitions between narrators")

if __name__ == "__main__":
    main()