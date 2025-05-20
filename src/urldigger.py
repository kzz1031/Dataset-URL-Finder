import re
from chat_manager import chat_inst

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
    possible_urls = list(set(possible_urls))
    possible_urls = [url for url in possible_urls if not url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    
    
    return high_confidence_urls, possible_urls


def extract_urls_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    
    high_confidence_urls, possible_urls = extract_urls(text)
    
    return high_confidence_urls, possible_urls


def filter_possible_urls(possible_urls):
    # Use chat_inst to filter possible URLs
    filtered_urls = []
    
    if not possible_urls:
        return filtered_urls
    
    # Process each URL individually
    for url in possible_urls:
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
                    print(f"URL validated: {url}")
                    break
                elif 'INVALID' == response_text.upper():
                    print(f"URL rejected: {url}")
                    break
        except Exception as e:
            print(f"Error validating URL '{url}': {e}")
    
    return filtered_urls


def dig_urls_from_file(file_path):
    high_confidence_urls, possible_urls = extract_urls_from_file(file_path)
    
    filtered_urls = filter_possible_urls(possible_urls)
    
    return high_confidence_urls + filtered_urls


def dig_urls_from_text(text):
    high_confidence_urls, possible_urls = extract_urls(text)
    
    filtered_urls = filter_possible_urls(possible_urls)
    
    return high_confidence_urls + filtered_urls


if __name__ == "__main__":
    filepath = 'src/output/attachment?id=aVh9KRZdRk&name=pdf/auto/attachment?id=aVh9KRZdRk&name=pdf.md'
    print(dig_urls_from_file(filepath))
