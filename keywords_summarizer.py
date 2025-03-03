import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import time 

# Set up your Gemini API key
GEMINI_API_KEY = ""

def fetch_abstract(arxiv_url):
    # Fetch the arXiv page content using requests
    response = requests.get(arxiv_url)
    if response.status_code != 200:
        return f"Error: Unable to fetch {arxiv_url}, status code: {response.status_code}"

    # Parse the HTML content of the arXiv page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the abstract section
    abstract_tag = soup.find('blockquote', class_='abstract mathjax')
    if abstract_tag:
        # Get the content of the abstract and ensure a space after the "Abstract:" label
        abstract_text = abstract_tag.text.strip()
        # Ensure "Abstract:" has a space
        if abstract_text.startswith("Abstract:"):
            abstract_text = abstract_text.replace("Abstract:", "Abstract: ")
        return abstract_text
    else:
        return "Error: Abstract not found."

def summarize_with_gemini(abstract_text):
    # Set up the API endpoint for Gemini
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY
    
    headers = {
        "Content-Type": "application/json",
    }

    # Create the payload to send the abstract to Gemini with a more focused prompt
    data = {
        "contents": [{
            "parts": [{
                "text": f"Summarize the following abstract in 1-2 simple sentences. Focus on what the authors did, why, and the results: \n\n{abstract_text}"
            }]
        }]
    }

    # Make the POST request to the Gemini API
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        # Extract the summary from the response
        result = response.json()
        try:
            # Access the correct keys in the response structure
            summary = result['candidates'][0]['content']['parts'][0]['text']
            return summary
        except KeyError as e:
            return f"KeyError: {e}, check the response structure."
    else:
        return f"Error: Unable to get response, status code: {response.status_code}"

def fetch_papers(keywords, start_date, end_date, max_results_per_keyword):
    papers = []
    keyword_totals = {}  # Dictionary to store total papers found for each keyword
    
    for keyword in keywords:
        # Construct the arXiv API query URL for each keyword to get the total count
        query = f'all:"{keyword}"'
        query_url = f"http://export.arxiv.org/api/query?search_query=({query})+AND+submittedDate:[{start_date}+TO+{end_date}]&start=0&max_results=0"
        
        # Fetch the total number of papers for this keyword
        response = requests.get(query_url)
        if response.status_code != 200:
            print(f"Error: Unable to fetch total papers for keyword '{keyword}', status code: {response.status_code}")
            continue

        # Parse the XML response to get the total number of papers
        root = ET.fromstring(response.content)
        total_results = root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults').text
        keyword_totals[keyword] = int(total_results)  # Store the total count

        # Now fetch the top N papers for this keyword
        query_url = f"http://export.arxiv.org/api/query?search_query=({query})+AND+submittedDate:[{start_date}+TO+{end_date}]&start=0&max_results={max_results_per_keyword}"
        response = requests.get(query_url)
        if response.status_code != 200:
            print(f"Error: Unable to fetch papers for keyword '{keyword}', status code: {response.status_code}")
            continue

        # Parse the XML response to get the papers
        root = ET.fromstring(response.content)
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            link = entry.find('{http://www.w3.org/2005/Atom}link[@title="pdf"]').attrib['href']
            papers.append({'title': title, 'summary': summary, 'link': link, 'keyword': keyword})
    
    # Print the total number of documents found for each keyword
    print("\nTotal documents found per keyword:")
    for keyword, total in keyword_totals.items():
        print(f"{keyword}: {total} documents (retrieved {min(total, max_results_per_keyword)})")
    
    return papers

# Open the result file to store the summaries
with open("result.txt", "w") as result_file:
    # Prompt for user input
    keywords = input("Enter keywords separated by commas: ").strip().split(',')
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    max_results_per_keyword = int(input("Enter the number of results per keyword: ").strip())

    # Fetch papers based on keywords, date range, and max results per keyword
    papers = fetch_papers(keywords, start_date, end_date, max_results_per_keyword)
    
    for paper in papers:
        print(f"Fetching abstract for: {paper['title']}")
        
        # Fetch the abstract
        abstract = paper['summary']
        if not abstract.startswith("Error"):
            # Summarize the abstract using Gemini
            summary = summarize_with_gemini(abstract)
            result_file.write(f"Keyword: {paper['keyword']}\nTitle: {paper['title']}\nLink: {paper['link']}\nSummary: {summary}\n\n")
            print(f"Summary for {paper['title']}:\n{summary}\n")
        else:
            result_file.write(f"Keyword: {paper['keyword']}\nTitle: {paper['title']}\nLink: {paper['link']}\nSummary: Error fetching abstract\n\n")
            print(f"Error fetching abstract for {paper['title']}\n")
        # Add a 1-second delay between each summary request
        time.sleep(2)