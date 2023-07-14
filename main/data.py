import json
import os
from tqdm import tqdm
from bs4 import BeautifulSoup

def create_and_format(data_folder):
    # Open the HTML file
    with open(os.path.join(data_folder, 'arxiv_papers.html'), 'r') as file:
        data = file.read()

    soup = BeautifulSoup(data, 'html.parser')
    tbody = soup.find('tbody')
    rows = tbody.find_all('tr')
    abstracts = []

    for row in tqdm(rows):
        abstract = row.find_all('td')[4].text

        formatted_text = f"### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis: {abstract}"

        json_data = {
            "text": formatted_text
        }

        abstracts.append(json_data)

    with open(os.path.join(data_folder, 'abstracts.json'), 'w') as json_file:
        json.dump(abstracts, json_file, indent=4)

if __name__ == "__main__":
    create_and_format('../data')