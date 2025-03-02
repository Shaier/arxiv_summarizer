# ArXiv Paper Summarizer

This repository provides a Python script to fetch and summarize research papers from arXiv using the free Gemini API. Additionally, it demonstrates how to automate the extraction and summarization of arXiv articles daily based on specific keywords (see the section titled "Automatic Daily Extraction and Summarization" below). The tool is designed to help researchers, students, and enthusiasts quickly extract key insights from arXiv papers without manually reading through lengthy documents.

## Features
- **Single Paper Summarization**: Summarize a single arXiv paper by providing its URL.
- **Batch Summarization**: Summarize multiple arXiv papers by listing their URLs in a text file.
- **Easy Setup**: Simple installation and configuration process using Conda and pip.
- **Gemini API Integration**: Leverages the free Gemini API for high-quality summarization.

## Prerequisites
- Python 3.11
- Conda (for environment management)
- A Gemini API key (free to obtain)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Shaier/arxiv_summarizer.git
cd arxiv_summarizer
```

### 2. Set Up the Conda Environment
Create and activate a Conda environment with Python 3.11:
```bash
conda create -n arxiv_summarizer python=3.11
conda activate arxiv_summarizer
```

### 3. Install Dependencies
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API Key
Obtain your Gemini API key from [Google's Gemini API page](https://ai.google.dev/gemini-api/docs/api-key). Once you have the key, open the `arxiv_summarize.py` file and replace `YOUR_GEMINI_API_KEY` on line 5 with your actual API key.

## Usage

### Summarize a Single Paper
To summarize a single arXiv paper, run the script and provide the arXiv URL (ensure it is the abstract page, not the PDF link):
```bash
python arxiv_summarize.py
```
When prompted:
1. Enter `1` to summarize a single paper.
2. Provide the arXiv URL (e.g., `https://arxiv.org/abs/2410.08003`).

### Summarize Multiple Papers
To summarize multiple papers:
1. Add the arXiv URLs to the `links.txt` file, with one URL per line.
2. Run the script:
```bash
python arxiv_summarize.py
```
3. When prompted, enter `2` to process all URLs listed in `links.txt`. The result summaries will be written in `result.txt`.

## Example
Here’s an example of how to use the script:
```bash
python arxiv_summarize.py
> Enter 1 for single paper or 2 for multiple papers: 1
> Enter the arXiv URL: https://arxiv.org/abs/2410.08003
```

## Automatic Daily Extraction and Summarization  
  
 You can automate the extraction and summarization of arXiv articles based on specific keywords using Google Apps Script.  
 This setup will run daily and add newly found article titles (with links and summaries) to a Google Doc.  
  
 ### Steps to Set Up  
  
 1. **Open Google Apps Script**  
    - Log in to your Google account and go to [Google Apps Script](https://script.google.com/home/my).  
    - Click on **"New project"** in the top left.  
  
 2. **Create a Google Doc**  
    - Open [Google Docs](https://docs.google.com).  
    - Click **Blank document** to create a new document.  
    - Copy the **document ID** from the URL.  
      - The ID is the long string in the document's URL, e.g., `123HEM4h5aQwygDk_A-xNaJ8CUoyMZTFsChyMk`.  
  
 3. **Copy and Modify the Script**  
    - Open the `daily_arxiv.txt` file in this repository.  
    - Copy and paste its content into the Google Apps Script editor.  
    - Locate the `var docId` in the script (around line 3) and replace it with the **Google Doc ID** from Step 2.  
    - Add your **Gemini API Key** around **line 81** (look for `var apiKey =`).
    - Locate `var keywords = [...]` around **line 4** and update it with your preferred keywords.  
  
 4. **Test the Script**  
    - Click the **Run** button at the top to execute the script (you might need to provide permissions).  
    - If everything works correctly, your Google Doc should now contain a list of arXiv article titles with links.  
  
 5. **Schedule Daily Execution**  
    - Click on the **clock icon** on the left (Triggers).  
    - Click **"Add trigger"** in the bottom right.  
    - Configure the trigger settings:  
      - **Function**: Select the main function from the dropdown.  
      - **Event Source**: Choose **Time-driven**.  
      - **Type**: Select **Day timer**.  
      - **Time Range**: Pick a time slot (e.g., midnight to 1 AM).  
      - **Notifications**: Enable email notifications if you want updates.  
    - Click **Save**.  
  
 Now, your script will automatically fetch and summarize new arXiv articles daily based on your chosen keywords!  

### Summarize an Entire Paper by Search Query Terms
Sumamrize your specified number of arxiv papers by search query. This summarizes the entire paper pdf rather than just the abstract:
```bash
python arxiv_suuarize_by_search_results.py -numSummaries 5 -prompt "Summarize this document in 3 sentences." -apiKey "YOUR_API_KEY" -query "Large Language Models"
```

You can also set environemnt variables rather than console variables if you don't want to enter variables every script call.

## Contributing
Contributions are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

## Support
If you encounter any issues or have questions, feel free to open an issue.
