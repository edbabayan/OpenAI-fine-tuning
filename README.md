# OpenAI Fine-Tuning

This repository is dedicated to fine-tuning OpenAI models, specifically the ChatGPT-4o-mini model, on medical data. The goal is to customize the model to understand medical reports from doctors and classify them into appropriate categories such as "Surgery" or "Cardiologist".

## Project Objective

We will fine-tune the ChatGPT-4o-mini model to receive a report from a doctor and classify it into categories based on the type of specialty, such as surgery or cardiology. The fine-tuning will involve processing medical data, extracting relevant features, and optimizing the model for accurate classification.

## Requirements
**OpenAI API Key**: To interact with the OpenAI models, you need to add your OpenAI API key to the project.
   
   - Please create a file named `openai_key.json` in the root directory of this repository.
   - The file should contain the API key in the following format:

   ```json
   {
     "key": "your_openai_api_key"
   }
