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
import random

# Check and download required NLTK resources
def download_nltk_resource(resource_id, resource_name):
    try:
        # Use the exact resource_id path for finding
        nltk.data.find(resource_id)
        print(f"NLTK '{resource_name}' resource found ({resource_id}).")
    except LookupError:
        print(f"NLTK '{resource_name}' resource not found ({resource_id}). Downloading '{resource_name}'...")
        # Use the resource_name for downloading
        nltk.download(resource_name, quiet=True)
        # Re-check after download attempt using the exact resource_id
        try:
            nltk.data.find(resource_id)
            print(f"NLTK '{resource_name}' resource downloaded successfully ({resource_id}).")
        except LookupError:
            print(f"Error: Failed to download or locate NLTK '{resource_name}' resource ({resource_id}).")
            print("Please check your internet connection or NLTK setup.")
            sys.exit(1) # Exit if essential resource is missing

# Download punkt (main tokenizer data)
download_nltk_resource('tokenizers/punkt', 'punkt')
# Download punkt_tab (specifically the English data, using the exact path from the error)
download_nltk_resource('tokenizers/punkt_tab/english/', 'punkt_tab')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

def load_and_split_text(file_path):
    """Load text file and split into paragraphs"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        # Remove very short paragraphs (likely formatting artifacts)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
        return paragraphs
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def average_sentence_length(text):
    """Calculate average sentence length in words"""
    sentences = sent_tokenize(text)
    if not sentences:
        return 0
    words_per_sentence = [len(word_tokenize(s)) for s in sentences]
    return sum(words_per_sentence) / len(sentences)

def lexical_diversity(text):
    """Calculate lexical diversity (type-token ratio)"""
    words = word_tokenize(text.lower())
    if not words:
        return 0
    return len(set(words)) / len(words)

def count_pronouns(text):
    """Count first-person and third-person personal pronouns."""
    first_person = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'}
    third_person = {'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'}
    
    words = word_tokenize(text.lower())
    
    first_person_count = sum(1 for word in words if word in first_person)
    third_person_count = sum(1 for word in words if word in third_person)
    
    # Return counts for both categories
    return first_person_count, third_person_count

def extract_features(paragraphs):
    """Extract stylometric features from each paragraph"""
    features = []
    feature_names = [
        'avg_sentence_length',
        'lexical_diversity',
        'word_count',
        'total_pronoun_ratio',
        'third_person_pronoun_ratio',
        'third_vs_first_person_ratio',
        'no_first_person_binary',
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
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        avg_sent_len = average_sentence_length(paragraph)
        lex_div = lexical_diversity(paragraph)
        words = word_tokenize(paragraph.lower())
        word_count = len(words)
        
        # Pronoun usage - Updated
        first_person_count, third_person_count = count_pronouns(paragraph)
        total_pronoun_count = first_person_count + third_person_count
        
        total_pronoun_ratio = total_pronoun_count / word_count if word_count > 0 else 0
        third_person_pronoun_ratio = third_person_count / word_count if word_count > 0 else 0
        
        # Calculate third vs first ratio, handle division by zero
        total_first_third = first_person_count + third_person_count
        third_vs_first_person_ratio = (third_person_count / total_first_third) if total_first_third > 0 else 0
        
        # Binary feature for absence of first person
        no_first_person_binary = 1 if first_person_count == 0 else 0
        
        # POS distribution using spaCy
        doc = nlp(paragraph)
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        
        noun_pct = pos_counts['NOUN'] / total_tokens if total_tokens > 0 else 0
        verb_pct = pos_counts['VERB'] / total_tokens if total_tokens > 0 else 0
        adj_pct = pos_counts['ADJ'] / total_tokens if total_tokens > 0 else 0
        adv_pct = pos_counts['ADV'] / total_tokens if total_tokens > 0 else 0
        det_pct = pos_counts['DET'] / total_tokens if total_tokens > 0 else 0
        
        avg_word_len = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        punct_count = sum(1 for token in doc if token.is_punct)
        punct_density = punct_count / word_count if word_count > 0 else 0
        
        sent_count = len(sent_tokenize(paragraph))
        
        question_marks = paragraph.count('?')
        exclamation_marks = paragraph.count('!')
        semicolons = paragraph.count(';')
        
        total_punct = punct_count if punct_count > 0 else 1
        
        question_ratio = question_marks / total_punct
        exclamation_ratio = exclamation_marks / total_punct
        semicolon_ratio = semicolons / total_punct
        
        # Combine all features
        feature_vector = [
            avg_sent_len,
            lex_div,
            word_count,
            total_pronoun_ratio,
            third_person_pronoun_ratio,
            third_vs_first_person_ratio,
            no_first_person_binary,
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

def cluster_paragraphs(features, max_clusters=10, n_clusters_override=None):
    """Cluster paragraphs using Gaussian Mixture Model. 
       If n_clusters_override is given, use that. Otherwise, determine optimal clusters via BIC."""
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    bic_scores = None
    n_components_range = None
    optimal_n_clusters = None

    if n_clusters_override is not None:
        print(f"Using specified number of clusters: {n_clusters_override}")
        optimal_n_clusters = n_clusters_override
        gmm = GaussianMixture(n_components=optimal_n_clusters, 
                              covariance_type='full', 
                              random_state=42,
                              n_init=10)
        gmm.fit(scaled_features)
        best_gmm = gmm
    else:
        # Determine optimal number of clusters using BIC
        n_components_range = range(2, max_clusters + 1)
        bic_scores = []
        models = []

        print(f"Testing cluster numbers from 2 to {max_clusters}...")
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, 
                                  covariance_type='full', 
                                  random_state=42,
                                  n_init=10)
            gmm.fit(scaled_features)
            models.append(gmm)
            bic_scores.append(gmm.bic(scaled_features))
            print(f"  Clusters: {n_components}, BIC: {bic_scores[-1]:.2f}")

        # Find the model with the lowest BIC
        if not bic_scores: # Handle case where max_clusters < 2
             raise ValueError("max_clusters must be at least 2 to use BIC method.")
        optimal_n_clusters_index = np.argmin(bic_scores)
        optimal_n_clusters = n_components_range[optimal_n_clusters_index]
        best_gmm = models[optimal_n_clusters_index]
        print(f"Optimal number of clusters based on BIC: {optimal_n_clusters}")

    # Get results from the best model
    labels = best_gmm.predict(scaled_features)
    probabilities = best_gmm.predict_proba(scaled_features)
    confidence = np.max(probabilities, axis=1)
    
    # Return BIC scores and range only if calculated
    return labels, probabilities, confidence, scaled_features, best_gmm, optimal_n_clusters, bic_scores, n_components_range

def visualize_elbow_plot(bic_scores, n_components_range, optimal_n_clusters, output_dir='output'):
    """Visualize the BIC scores to show the elbow."""
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, bic_scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (Narrators)', fontsize=12)
    plt.ylabel('BIC Score (Lower is Better)', fontsize=12)
    plt.title('BIC Scores for Different Numbers of Clusters', fontsize=16)
    plt.xticks(n_components_range)
    # Highlight the chosen optimal number
    plt.axvline(optimal_n_clusters, color='r', linestyle='--', label=f'Optimal: {optimal_n_clusters}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'elbow_plot_bic.png'), dpi=300)
    plt.close()
    print(f"Saved BIC elbow plot to {os.path.join(output_dir, 'elbow_plot_bic.png')}")

def find_ambiguous_sections(paragraphs, labels, confidence, probabilities, threshold=0.7):
    """Identify paragraphs with ambiguous narrator assignment"""
    ambiguous_paragraphs = []
    
    for i, conf in enumerate(confidence):
        if conf < threshold:
            ambiguous_paragraphs.append({
                'index': i,
                'text': paragraphs[i],
                'confidence': float(conf),
                'assigned_cluster': int(labels[i]),
                'probabilities': probabilities[i].tolist()
            })
    
    return ambiguous_paragraphs

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
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
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
    plt.xlabel('Paragraph Index', fontsize=12)
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
    plt.figure(figsize=(12, 8))
    sns.heatmap(means, annot=False, cmap='coolwarm', 
                xticklabels=feature_names, 
                yticklabels=[f'Narrator {i}' for i in range(means.shape[0])])
    plt.title('Feature Importance by Narrator Cluster', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()

def save_results(paragraphs, labels, confidence, probabilities, ambiguous_sections, 
                transitions, output_dir='output'):
    """Save analysis results to files"""
    # Save paragraph assignments
    results = []
    for i, paragraph in enumerate(paragraphs):
        results.append({
            'paragraph_index': i,
            'text': paragraph[:100] + '...' if len(paragraph) > 100 else paragraph,
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

def generate_summary_report(paragraphs, labels, confidence, ambiguous_sections, 
                           transitions, n_clusters, was_bic_used, output_dir='output'):
    """Generate a summary report of findings"""
    total_paragraphs = len(paragraphs)
    ambiguous_count = len(ambiguous_sections)
    transition_count = len(transitions)
    
    avg_confidence = np.mean(confidence)
    narrator_distribution = pd.Series(labels).value_counts().sort_index()
    
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("=== NARRATOR ANALYSIS SUMMARY ===\n\n")
        f.write(f"Total paragraphs analyzed: {total_paragraphs}\n")
        if was_bic_used:
            f.write(f"Optimal number of narrators (determined by BIC): {n_clusters}\n")
        else:
            f.write(f"Specified number of narrators: {n_clusters}\n")
        f.write(f"Average narrator confidence: {avg_confidence:.2f}\n\n")
        
        f.write("Narrator distribution:\n")
        for narrator in range(n_clusters):
            count = narrator_distribution.get(narrator, 0)
            percentage = (count / total_paragraphs) * 100 if total_paragraphs > 0 else 0
            f.write(f"  Narrator {narrator}: {count} paragraphs ({percentage:.1f}%)\n")
        
        f.write(f"\nAmbiguous narrator sections: {ambiguous_count} ({(ambiguous_count/total_paragraphs)*100:.1f}%)\n")
        f.write(f"Narrator transitions: {transition_count}\n\n")
        
        f.write("=== TOP AMBIGUOUS SECTIONS ===\n\n")
        top_ambiguous = sorted(ambiguous_sections, key=lambda x: x['confidence'])[:5]
        for i, section in enumerate(top_ambiguous):
            f.write(f"Ambiguous Section {i+1} (Confidence: {section['confidence']:.2f}):\n")
            f.write(f"  Paragraph Index: {section['index']}\n")
            f.write(f"  Assigned Narrator: {section['assigned_cluster']}\n")
            f.write(f"  Text: {section['text'][:200]}...\n\n")
        
        f.write("=== KEY NARRATOR TRANSITIONS ===\n\n")
        key_transitions = sorted(transitions, key=lambda x: x['confidence_after'])[:5]
        for i, trans in enumerate(key_transitions):
            f.write(f"Transition {i+1} (Confidence After: {trans['confidence_after']:.2f}):\n")
            f.write(f"  Position: Paragraph {trans['position']}\n")
            f.write(f"  From Narrator {trans['from']} to Narrator {trans['to']}\n")
            f.write(f"  Confidence Before: {trans['confidence_before']:.2f}\n\n")

def generate_example_sentences(paragraphs, labels, confidence, n_examples=10, min_confidence=0.9, output_dir='output'):
    """Generate a file with example sentences for each narrator cluster, 
       selecting randomly from high-confidence sentences for better spread."""
    narrator_sentences = {i: [] for i in np.unique(labels)}
    
    # Collect all sentences meeting the minimum confidence threshold
    for i, paragraph in enumerate(paragraphs):
        label = labels[i]
        conf = confidence[i]
        
        if conf >= min_confidence:
            sentences = sent_tokenize(paragraph)
            # Store sentence, confidence, and paragraph index
            narrator_sentences[label].extend([(sent, conf, i) for sent in sentences if len(word_tokenize(sent)) > 5]) # Basic length filter

    output_path = os.path.join(output_dir, 'narrator_examples.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== EXAMPLE SENTENCES FOR EACH NARRATOR ===\n")
        f.write("(Randomly selected from sentences with confidence >= ")
        f.write(f"{min_confidence})\n\n")
        
        for narrator in sorted(narrator_sentences.keys()):
            f.write(f"--- Narrator {narrator} ---\n")
            
            high_conf_sentences = narrator_sentences[narrator]
            
            # Randomly sample if more examples exist than needed, otherwise take all
            if len(high_conf_sentences) > n_examples:
                selected_examples = random.sample(high_conf_sentences, n_examples)
            else:
                selected_examples = high_conf_sentences
            
            # Sort the selected examples by paragraph index for readability in the output file
            selected_examples.sort(key=lambda x: x[2]) 

            unique_sentences_added = set()
            examples_added = 0
            
            for sentence, conf, p_idx in selected_examples:
                # Double check for uniqueness, although random.sample should handle it
                if sentence not in unique_sentences_added:
                    f.write(f"  (Conf: {conf:.2f}, Para: {p_idx}) {sentence}\n")
                    unique_sentences_added.add(sentence)
                    examples_added += 1
            
            if examples_added == 0:
                 f.write("  (No high-confidence sentences found for this narrator)\n")
            elif examples_added < n_examples and len(high_conf_sentences) >= examples_added:
                 # Note if fewer examples were found than requested
                 if len(high_conf_sentences) < n_examples:
                     f.write(f"  (Only found {len(high_conf_sentences)} high-confidence sentences in total)\n")
                 else:
                     # This case might occur if random sample picked duplicates somehow, though unlikely
                     f.write(f"  (Selected {examples_added} unique sentences)\n")

            f.write("\n")
            
    print(f"Saved example sentences to {output_path}")

def generate_uncertain_sentences(paragraphs, labels, confidence, n_examples=100, output_dir='output'):
    """Generate a file listing the most uncertain sentences based on cluster confidence."""
    all_sentences_with_confidence = []

    # Iterate through paragraphs and their confidence scores
    for i, paragraph in enumerate(paragraphs):
        conf = confidence[i]
        label = labels[i]
        sentences = sent_tokenize(paragraph)
        
        # Assign the paragraph's confidence to each sentence within it
        for sentence in sentences:
            # Store sentence, confidence, paragraph index, and assigned label
            if len(word_tokenize(sentence)) > 3: # Basic filter for very short/fragmented sentences
                all_sentences_with_confidence.append({
                    'sentence': sentence,
                    'confidence': conf,
                    'paragraph_index': i,
                    'assigned_narrator': label
                })

    # Sort all sentences by confidence (ascending - least confident first)
    sorted_sentences = sorted(all_sentences_with_confidence, key=lambda x: x['confidence'])

    # Select the top N most uncertain sentences
    uncertain_examples = sorted_sentences[:n_examples]

    # Write to file
    output_path = os.path.join(output_dir, 'uncertain_sentences.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"=== TOP {n_examples} MOST UNCERTAIN SENTENCES ===\n")
        f.write("(Sorted by lowest confidence score)\n\n")
        
        for example in uncertain_examples:
            f.write(f"Confidence: {example['confidence']:.4f}\n")
            f.write(f"Assigned Narrator: {example['assigned_narrator']}\n")
            f.write(f"Paragraph Index: {example['paragraph_index']}\n")
            f.write(f"Sentence: {example['sentence']}\n\n")
            
    print(f"Saved {len(uncertain_examples)} most uncertain sentences to {output_path}")

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
    if not check_dependencies():
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description='Analyze narrators in a text using unsupervised learning.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--clusters', '-c', type=int, default=None, help='Specify the exact number of narrator clusters (overrides BIC method)')
    parser.add_argument('--max-clusters', '-mc', type=int, default=10, help='Maximum number of clusters to test for BIC elbow method (used if --clusters is not set)')
    parser.add_argument('--threshold', '-t', type=float, default=0.7, 
                        help='Confidence threshold for ambiguous narrator detection')
    parser.add_argument('--output', '-o', type=str, default='output', 
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    print("Loading and processing text...")
    paragraphs = load_and_split_text(args.input)
    
    if not paragraphs:
        print("No paragraphs found. Check the file path and format.")
        return
    
    print(f"Extracted {len(paragraphs)} paragraphs.")
    
    print("Extracting stylometric features...")
    features, feature_names = extract_features(paragraphs)
    
    print("Clustering paragraphs...")
    # Pass command-line arguments to the clustering function
    labels, probabilities, confidence, scaled_features, model, n_clusters, bic_scores, n_components_range = cluster_paragraphs(
        features, 
        n_clusters_override=args.clusters, # Use value from command line (or None)
        max_clusters=args.max_clusters    # Use value from command line (or default)
    )
    print(f"Clustering complete. Using {n_clusters} clusters.")

    # Determine if BIC was used (i.e., if clusters were not overridden)
    was_bic_used = args.clusters is None

    # Visualize elbow plot only if BIC was calculated
    if was_bic_used and bic_scores is not None and n_components_range is not None:
        print("Visualizing BIC elbow plot...")
        visualize_elbow_plot(bic_scores, n_components_range, n_clusters, args.output)
    elif not was_bic_used:
        print("Skipping BIC elbow plot visualization as cluster count was specified.")
    else:
        print("Skipping BIC elbow plot visualization (BIC scores not available).")
    
    print("Finding ambiguous narrator sections...")
    ambiguous_sections = find_ambiguous_sections(paragraphs, labels, confidence, 
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
    # Pass the correct flag indicating if BIC was used
    generate_summary_report(paragraphs, labels, confidence, ambiguous_sections, 
                           transitions, n_clusters, was_bic_used, args.output)
    
    print("Saving detailed results...")
    save_results(paragraphs, labels, confidence, probabilities, 
                ambiguous_sections, transitions, args.output)

    print("Generating example sentences for each narrator...")
    generate_example_sentences(paragraphs, labels, confidence, n_examples=10, output_dir=args.output)

    print("Generating list of most uncertain sentences...")
    generate_uncertain_sentences(paragraphs, labels, confidence, n_examples=100, output_dir=args.output)
    
    print("\nAnalysis complete! Check the output files in the", args.output, "directory:")
    print("- narrator_clusters.png: Visual representation of narrator clusters")
    print("- narrator_timeline.png: Narrator transitions throughout the text")
    print("- confidence_distribution.png: Distribution of confidence scores by narrator")
    print("- feature_importance.png: Importance of different features for each narrator")
    print("- summary_report.txt: Summary of key findings")
    print("- narrator_analysis_results.csv: Complete analysis results")
    print("- ambiguous_narrators.csv: Paragraphs with ambiguous narrator assignment")
    print("- narrator_transitions.csv: Transitions between narrators")
    print("- narrator_examples.txt: Example sentences for each narrator")
    print("- uncertain_sentences.txt: The 100 sentences with the lowest confidence scores")

if __name__ == "__main__":
    main()