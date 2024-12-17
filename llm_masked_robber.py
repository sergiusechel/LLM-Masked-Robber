'''
Disclaimer: This tool is intended solely for academic and informational purposes. The analysis and descriptions of prompt injection techniques and related adversarial testing methods are provided to understand potential vulnerabilities in large language models (LLMs) and to advance the field of cybersecurity. Under no circumstances should the techniques described be used to exploit, manipulate, or compromise LLMs or other artificial intelligence systems outside of controlled, authorized research environments. All testing was conducted ethically, and with the aim of responsibly disclosing potential issues to improve the resilience and security of AI systems. The author does not assume any responsibility for misuse of the information presented.
'''

import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn.functional as F
import nltk
from nltk import pos_tag
import ssl

# Bypass SSL verification for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Load the tokenizer and model from local cache
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir="roberta-base", local_files_only=True)
model = RobertaForMaskedLM.from_pretrained("roberta-base", cache_dir="roberta-base", local_files_only=True)

def get_user_inputs():
    """Prompt the user for sentence input and top-k predictions."""
    masked_sentence = input("Type in your sentence and use __ to mask keywords: ")
    while True:
        try:
            top_k_input = int(input("Enter the number of top predictions to retrieve for each masked token: "))
            if top_k_input <= 0:
                raise ValueError("Top-k must be a positive integer.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")
    return masked_sentence, top_k_input

def is_noun_or_verb(word):
    """
    Check if a word is a noun or verb using POS tagging.
    
    Args:
    - word (str): The word to be checked.
    
    Returns:
    - (bool): True if the word is a noun or verb, False otherwise.
    """
    pos = pos_tag([word])[0][1]  # POS tagging for the word
    return pos.startswith('N') or pos.startswith('V')  # Noun (NN) or Verb (VB)

def mask_and_predict_top_k_with_probs(text, top_k=5, masked_token="__"):
    """
    Mask tokens in text and return the top-k predicted words with probabilities,
    prioritizing nouns and verbs.
    """
    text_with_mask = text.replace(masked_token, tokenizer.mask_token)
    input_ids = tokenizer.encode(text_with_mask, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits
    
    masked_indexes = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    predictions_dict = {}
    
    for idx in masked_indexes:
        logits = predictions[0, idx]
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_token_ids = probs.topk(top_k * 2)  # Retrieve more predictions to filter later
        top_k_tokens = [tokenizer.decode([token_id]).strip() for token_id in top_k_token_ids.tolist()]
        
        # Filter tokens to prioritize nouns and verbs
        filtered_tokens = [(word, prob) for word, prob in zip(top_k_tokens, top_k_probs.tolist()) if is_noun_or_verb(word)]
        
        # If we don't have enough valid tokens, fall back to original predictions
        if len(filtered_tokens) < top_k:
            filtered_tokens.extend(zip(top_k_tokens, top_k_probs.tolist()))
        
        # Limit to top-k
        predictions_dict[f"Masked Position {idx.item()}"] = filtered_tokens[:top_k]
    
    return predictions_dict

def generate_prompts_with_predictions(text, predictions_dict, masked_token="__"):
    """
    Generate prompts by replacing masked tokens with top predictions.
    """
    prompt_list = [text]
    for _, predictions in predictions_dict.items():
        new_prompt_list = []
        for prompt in prompt_list:
            for word, _ in predictions:
                new_prompt = prompt.replace(masked_token, word, 1)
                new_prompt_list.append(new_prompt)
        prompt_list = new_prompt_list
    return prompt_list

def save_predictions_to_file(masked_sentence, predictions_dict, prompt_variations):
    """Save predictions and generated prompts to files."""
    with open('adversarial_prompts_masked_tokens.csv', 'a', encoding='utf-8') as f:
        for position, predictions in predictions_dict.items():
            for word, prob in predictions:
                f.write(f"{masked_sentence},{position},{word},{prob:.4f}\n")

    with open('adversarial_prompts.csv', 'a', encoding='utf-8') as f:
        for prompt in prompt_variations:
            f.write(f"{prompt}\n")

def main():
    """Main function to execute the script."""
    masked_sentence, top_k_input = get_user_inputs()
    if "__" not in masked_sentence:
        print("Error: Your sentence must contain at least one '__' token.")
        return
    
    # Predict words and probabilities
    predicted_words_with_probs = mask_and_predict_top_k_with_probs(masked_sentence, top_k=top_k_input)
    
    # Print predictions
    print(f"\nTop {top_k_input} Predictions with Probabilities for Each Masked Token:")
    for position, predictions in predicted_words_with_probs.items():
        print(f"{position}:")
        for word, prob in predictions:
            print(f"  {word}: {prob:.4f}")
    
    # Generate and print prompt variations
    prompt_variations = generate_prompts_with_predictions(masked_sentence, predicted_words_with_probs)
    print("\nGenerated Prompts with All Predicted Words:")
    for prompt in prompt_variations:
        print(prompt)
    
    # Save outputs to files
    save_predictions_to_file(masked_sentence, predicted_words_with_probs, prompt_variations)
    print("\nResults have been saved to 'adversarial_prompts_masked_tokens.csv' and 'adversarial_prompts.csv'.")

if __name__ == "__main__":
    main()
