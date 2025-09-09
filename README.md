
DeepStock: An LLM for Stock Analysis
DeepStock is a comprehensive project for building and training a Large Language Model (LLM) specialized for stock analysis tasks. The pipeline covers everything from collecting and processing a large-scale dataset, creating specialized prompts, fine-tuning a pre-trained LLM, and finally, generating sample analyses.

Due to data limitations, this project primarily relies on open APIs like Yahoo Finance and other publicly available datasets.

✨ Key Features
📊 Automated Data Collection: Automatically fetches historical stock prices, company information, and financial statements from Yahoo Finance.

📰 News Aggregation: Incorporates financial news headlines from a massive open-source Kaggle dataset.

⚙️ Efficient Data Processing: Efficiently processes and merges data from various sources, using a caching mechanism to avoid redundant downloads.

🤖 Specialized Prompt Creation: Automatically formats the processed data into a structured system-user-assistant prompt format for training.

🧠 Model Fine-tuning: Fine-tunes the deepseek-coder-6.7b-instruct model using the trl library and SFTTrainer on the custom-built dataset.

☁️ Cloud Scalability: Integrates with Modal to run the data collection process in a distributed cloud environment, significantly speeding it up.

🤗 Hugging Face Integration: Automatically uploads the final processed dataset and the trained model to the Hugging Face Hub for easy access and sharing.

⚙️ Workflow Pipeline
The project's workflow is divided into four main stages:

1. Data Preparation
Downloads the list of S&P 500 companies.

Collects news, financial statements, and historical price data for each company.

Combines all information into a single dataset and uploads it to the Hugging Face Hub.

2. Prompt Engineering
The raw data is converted into structured, conversational prompts to teach the model how to answer analytical questions.

Example Prompt Structure:

<|im_start|>system
A conversation between User and Assistant.
<|im_end|>
<|im_start|>user
{Stock-related question}
<|im_end|>
<|im_start|>assistant
{Analytical answer}
<|im_end|>
3. Model Training
Loads the base model deepseek-coder-6.7b-instruct.

Uses the SFTTrainer (Supervised Fine-tuning Trainer) from the trl library to train the model on the prompt-formatted dataset.

Training parameters like learning rate, batch size, and the number of epochs are configured to optimize performance.

4. Inference
After training, the model can be used to answer new, unseen questions.

Example Model Output:

User: Is 3M stock good?

Model: To determine if 3M stock is good, we need to consider various factors such as its financial health, market trends, industry position, and future growth prospects. As of now, 3M has a market capitalization of over $77 billion... Its financial health is strong, and it has a history of steady dividend payments... The overall trend in the stock market suggests that it is a good time to purchase 3M stock...

🛠️ Technology Stack
Core: Python, Jupyter Notebook

Data Handling: Pandas, yfinance

LLM & Training: transformers, peft, trl, bitsandbytes

Cloud Computing: Modal

Platforms: Hugging Face Hub, Wandb (Weights & Biases)

🚀 Usage Guide
1. Set Up Credentials
Add your HUGGING_FACE_HUB_TOKEN and WANDB_API_KEY as secrets in your environment (e.g., Google Colab secrets).

Configure your Modal token.

2. Run the Notebook
Execute the cells in the Stock_DeepSeek_With_Dataset_Preparation_V1.ipynb notebook.

The notebook will run the entire pipeline, from data preparation and prompt creation to model training and a final inference example.

💡 Tip: If you want to skip the time-consuming data preparation step, you can download the pre-processed dataset directly from here:
chuotchuilacduong/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2

3. Access the Artifacts
The final dataset and the trained model will be available in your Hugging Face Hub repository.

⚠️ Disclaimer
This project is for educational and research purposes only. The data is collected from open sources and may contain inaccuracies. This is not financial advice. Do not use the dataset or models trained from this project to make actual investment decisions.
