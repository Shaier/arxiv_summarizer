import arxiv
from google import genai
from google.genai import types
import httpx
import os
import argparse
import sys

# Replace this with your API Key if you wish to insert it directly into the script rather than 
# via environment variable or command line. Otherwise set API_KEY environment variable or 
# pass in console var -apiKey. 
# Likewise specify a QUERY env var or -query console var for search terms
DEFAULT_API_KEY = None
parser = argparse.ArgumentParser(description="Arxive Summarizer by Search Results")

def set_console_variables():
    parser.add_argument('-numSummaries', help="The number of documents to gather and summarize from search results")
    parser.add_argument('-prompt', help="The prommpt to use for model summarization")
    parser.add_argument('-apiKey', help="Insert your API Key here if you prefer to pass as a console var")
    parser.add_argument('-query', help="Insert a search query for arxiv results")

# Try and use an environment prompt variable
# If a console variable is set, override environment variable with console var
# Otherwise go to a default value
def set_var(env_var_name, console_var, default_val, dtype=None):
    var = os.environ.get(env_var_name)

    if console_var != None:
        var = console_var
    elif var == None:
        var = default_val

    # Try to convert to specified dtype
    if dtype is not None:
        try:
            return dtype(var)
        except (ValueError, TypeError) as e:
            print(f"Error converting {var} to {dtype}: {e}")
            # fall back to default_val
            return default_val

    return var

def generate_summary(client, doc_url, prompt):
        # Retrieve and encode the PDF byte
        doc_data = httpx.get(doc_url).content

        response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            types.Part.from_bytes(
                data=doc_data,
                mime_type='application/pdf',
            ),
            prompt])
        
        return response.text

def main():
    set_console_variables()
    args = parser.parse_args(sys.argv[1:])

    # Set global vars
    NUM_SUMMARIES = set_var("NUM_SUMMARIES", args.numSummaries, 1, int)
    PROMPT = set_var("PROMPT", args.prompt, "Create a short summary for this document. Focus on what the authors did, why, and the results.")
    API_KEY = set_var("API_KEY", args.apiKey, DEFAULT_API_KEY)
    SEARCH_QUERY = set_var("QUERY", args.query, "Large Language Models")

    if API_KEY == None:
        print("Please specify an API Key via the environment variable API_KEY or console -apiKey")
        return

    client = genai.Client(api_key=API_KEY)

    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=NUM_SUMMARIES
    )

    with open("result.txt", "w") as result_file:
        for result in arxiv.Client().results(search):
            summary = generate_summary(client, result.pdf_url, PROMPT)

            result_file.write(f"arXiv URL: {result.pdf_url}\nSummary: {summary}\n\n")
            print(f"Summary for {result.pdf_url}:\n{summary}\n")

if __name__ == '__main__':
    main()