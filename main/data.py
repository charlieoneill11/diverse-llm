import json
import os
from tqdm import tqdm
from bs4 import BeautifulSoup

import os
import json
import openai

openai.api_key = "22423a0765d0440b956f5efa647427b2" #os.getenv("AZURE_OPENAI_KEY")
openai.api_base = "https://charlie-resouruce.openai.azure.com/" # os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future

deployment_name='beast' # This will correspond to the custom name you chose for your deployment when you deployed a model. 


### ABSTRACTS DATASET ###

def create_and_format_abstracts(data_folder):
    # Open the HTML file
    with open(os.path.join(data_folder, 'arxiv_papers.html'), 'r') as file:
        data = file.read()

    rows = BeautifulSoup(data, 'html.parser').find('tbody').find_all('tr')
    abstracts = []

    for row in tqdm(rows):
        abstract = row.find_all('td')[4].text

        formatted_text = f"### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis: {abstract}"

        json_data = {"text": formatted_text}

        abstracts.append(json_data)

    with open(os.path.join(data_folder, 'abstracts.json'), 'w') as json_file:
        json.dump(abstracts, json_file, indent=4)

### HYPOTHESES DATASET ###

# Define the prompt
prompt = 'Extract the scientific hypothesis presented in this abstract: \
          "We search for evidence of dark matter in the Milky Way by utilizing the \
          stellar number density distribution and kinematics measured by the Sloan\
           Digital Sky Survey (SDSS) to heliocentric distances exceeding ~10 kpc. We\
           nemploy the cylindrically symmetric form of Jeans equations and focus on the\
           morphology of the resulting acceleration maps, rather than the normalization of\
           the total mass as done in previous, mostly local, studies. Jeans equations are\
           first applied to a mock catalog based on a cosmologically derived N-body + SPH\
           simulation, and the known acceleration (gradient of gravitational potential) is successfully recovered. The same simulation is also used to quantify the impact\\nof dark matter on the total acceleration. We use Galfast, a code designed to\\nquantitatively reproduce SDSS measurements and selection effects, to generate a\\nsynthetic stellar catalog. We apply Jeans equations to this catalog and produce\\ntwo-dimensional maps of stellar acceleration. These maps reveal that in a\\nNewtonian framework, the implied gravitational potential cannot be explained by\\nvisible matter alone. The acceleration experienced by stars at galactocentric\\ndistances of ~20 kpc is three times larger than what can be explained by purely\\nvisible matter. The application of an analytic method for estimating the dark\\nmatter halo axis ratio to SDSS data implies an oblate halo with q_DM = 0.47 +/-\\n0.14 within the same distance range. These techniques can be used to map the\\ndark matter halo to much larger distances from the Galactic center using\\nupcoming deep optical surveys, such as LSST." Present the hypothesis in one sentence in the form of a question that the paper aims to answer. Answer in the form of HYPOTHESIS: ...\
           HYPOTHESIS:'

# Define a function that uses chat completion to extract the hypothesis
def extract_hypothesis(abstract):
    
    # Send a completion call to generate an answer
    completion = openai.ChatCompletion.create(
      engine=deployment_name,
      messages=[
        {"role": "system", "content": "You are an expert astronomer that excels at extracting hypotheses from the abstracts of scientific papers."},
        {"role": "user", "content": "Extract the scientific hypothesis presented in this abstract: "+abstract+"\n"+"Present the hypothesis in one sentence in the form of a question that the paper aims to answer. HYPOTHESIS:"}
      ],
      max_tokens=200,
      stop='?',
    )
    # Return the hypothesis
    return completion.choices[0].message.content+'?'

# Create a function that batches the above over the whole dataset
# We should only look at everything that comes after Hypothesis:

def batch_extract_hypothesis(data):
  
  print(f'Extracting hypotheses from {len(data)} abstracts...')
  
  # Initialise a list to store the extracted hypotheses
  hypotheses = []

  # Loop over the data
  for i in tqdm(range(len(data))):
      
      # Extract the hypothesis
      hypothesis = extract_hypothesis(data[i]['text'].split('Hypothesis: ')[1])
      
      # Append to the list
      hypotheses.append(hypothesis)

  return hypotheses

def create_and_format_hypotheses(data_folder, offset=0):
    print(f'Creating and formatting hypotheses...')
    
    # Open the JSON file
    with open(os.path.join(data_folder, 'abstracts.json'), 'r') as file:
        data = json.load(file)
    
    print(f'Extracting hypotheses from {len(data[offset:])} abstracts...')
    
    # Check if hypotheses.json already exists
    if os.path.exists(os.path.join(data_folder, 'hypotheses.json')):
        # If it exists, we open it in read mode to check its content
        with open(os.path.join(data_folder, 'hypotheses.json'), 'r') as json_file:
            # If file is not empty, we load the existing data
            content = json_file.read()
            if content:
                hypotheses_json = json.loads(content)
            else:  # If file is empty, initialize an empty list
                hypotheses_json = []
    else:
        # If it does not exist, initialize an empty list
        hypotheses_json = []
        
    # Loop over the data
    for i in tqdm(range(len(data[offset:]))):
        # Extract the hypothesis
        hypothesis = extract_hypothesis(data[offset+i]['text'].split('Hypothesis: ')[1])
        
        # Format the text
        formatted_text = f"### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis: {hypothesis}"
        
        # Append to the list
        hypotheses_json.append({'text': formatted_text})
        
        # Open the output file in write mode and dump the updated list
        with open(os.path.join(data_folder, 'hypotheses.json'), 'w') as out_file:
            json.dump(hypotheses_json, out_file, indent=4)

if __name__ == "__main__":
    # offset will be number of entries in hypotheses.json file

    # read in file
    try:
        current_hypotheses = json.load(open('../data/hypotheses.json', 'r'))
        offset = len(current_hypotheses)
    except: offset = 0
    print(f"Offset = {offset}")
    create_and_format_hypotheses('../data', offset=offset)