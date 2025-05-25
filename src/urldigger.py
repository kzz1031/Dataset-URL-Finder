import re
import os
from chat_manager import chat_inst
from tqdm import tqdm

def extract_urls(text):
    # Regular expression for high confidence URLs (with http/https or www prefix)
    high_confidence_pattern = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    
    # Regular expression for possible URLs (domain-like patterns without http/https or www prefix)
    possible_url_pattern = r'([a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,})'
    
    # Find matches for both patterns
    high_confidence_urls = re.findall(high_confidence_pattern, text)
    possible_urls_all = re.findall(possible_url_pattern, text)
    
    # Filter out URLs from possible_urls that are already in high_confidence_urls
    possible_urls = [url for url in possible_urls_all if url not in high_confidence_urls]
    possible_urls = [url.strip() for url in possible_urls if url.strip()]
    possible_urls = [url.strip(',') for url in possible_urls if url.strip()]
    possible_urls = [url.strip(')') for url in possible_urls if url.strip()]
    possible_urls = [url.strip('.') for url in possible_urls if url.strip()]
    possible_urls = [url.strip('"') for url in possible_urls if url.strip()]
    possible_urls = list(set(possible_urls))
    possible_urls = [url for url in possible_urls if not url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', ')'))]
    # not matching \d+\.\d+
    possible_urls = [url for url in possible_urls if not re.match(r'\d+\.\d+', url)]
    # not matching ^.*\.$
    possible_urls = [url for url in possible_urls if not re.match(r'^.*\.$', url)]
    # not matching ^.*\."$
    # possible_urls = [url for url in possible_urls if not re.match(r'^.*\."$', url)]
    # not matching ^.*\.jpg"$
    # possible_urls = [url for url in possible_urls if not re.match(r'^.*\.jpg"$', url)]
    
    return high_confidence_urls, possible_urls


def extract_urls_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    
    high_confidence_urls, possible_urls = extract_urls(text)
    
    return high_confidence_urls, possible_urls


def filter_possible_urls(possible_urls):
    # Use chat_inst to filter possible URLs
    print(possible_urls)
    filtered_urls = []
    
    if not possible_urls:
        return filtered_urls
    
    # Process each URL individually
    print('Validating possible URLs...')
    for url in tqdm(possible_urls):
        # Create a prompt for the AI to validate a single URL
        prompt = f"""Is this string a valid URL that may contain a dataset and can be directly accessed by the browser? The string is '{url}'.
        Please respond with just 'VALID' or 'INVALID' and no other explanation."""
        
        try:
            while True:
                # Query the AI for this specific URL
                response = chat_inst.invoke(prompt)
                response_text = response.content
                
                # Check if the response indicates the URL is valid
                if 'VALID' == response_text.upper():
                    filtered_urls.append(url)
                    # print(f"URL validated: {url}")
                    break
                elif 'INVALID' == response_text.upper():
                    # print(f"URL rejected: {url}")
                    break
        except Exception as e:
            print(f"Error validating URL '{url}': {e}")
    
    return filtered_urls


def dig_urls_from_file(file_path):
    high_confidence_urls, possible_urls = extract_urls_from_file(file_path)
    
    filtered_urls = filter_possible_urls(possible_urls)
    
    return high_confidence_urls + filtered_urls


def gather_texts(outdir, mdname):
    text = ''
    with open(outdir + '/' + mdname + '.md', 'r') as file:
        text = file.read()
    if os.path.exists(outdir + '/' + mdname + '_middle.json'):
        with open(outdir + '/' + mdname + '_middle.json', 'r') as file:
            text += file.read()
    if os.path.exists(outdir + '/' + mdname + '_content_list.json'):
        with open(outdir + '/' + mdname + '_content_list.json', 'r') as file:
            text += file.read()
    if os.path.exists(outdir + '/' + mdname + '_content.json'):
        with open(outdir + '/' + mdname + '_model.json', 'r') as file:
            text += file.read()
    return text

def dig_urls_from_text(texts):
    high_confidence_urls, possible_urls = extract_urls(texts)
    
    filtered_urls = filter_possible_urls(possible_urls)
    
    return high_confidence_urls + filtered_urls


def dig_context_of_urls(text, urls):
    from config import CONTEXT_LENGTH  # Context Window Length: 2 * CONTEXT_LENGTH
    
    url_context_dict = {}
    
    for url in urls:
        # Find all occurrences of the URL in the text
        url_positions = [m.start() for m in re.finditer(re.escape(url), text)]
        
        contexts = []
        for pos in url_positions:
            # Calculate start and end positions for the context window
            start_pos = max(0, pos - CONTEXT_LENGTH)
            end_pos = min(len(text), pos + len(url) + CONTEXT_LENGTH)
            
            # Extract the context
            context = text[start_pos:end_pos]
            
            # Add markers to show where the URL is in the context
            if start_pos > 0:
                context = "..." + context
            if end_pos < len(text):
                context = context + "..."
                
            contexts.append(context)
        
        # Store all contexts for this URL
        if contexts:
            url_context_dict[url] = contexts
    
    return url_context_dict

if __name__ == "__main__":
    outdir = 'output/C4NbtYnyQg/auto'
    mdname = 'C4NbtYnyQg'
    texts = gather_texts(outdir, mdname)
    urls = dig_urls_from_text(texts)
    url_to_context = dig_context_of_urls(texts, urls)
    from pprint import pprint
    pprint(url_to_context)
