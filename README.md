# DeepStock: An LLM for Stock Analysis

**DeepStock** is a comprehensive project for building and training a **Large Language Model (LLM)** specialized for stock analysis tasks.  
The pipeline covers everything from collecting and processing a large-scale dataset, creating specialized prompts, fine-tuning a pre-trained LLM, and finally, generating sample analyses.

> ⚠️ Due to data limitations, this project primarily relies on open APIs like Yahoo Finance and other publicly available datasets.

---

## ✨ Key Features

- 📊 **Automated Data Collection**: Automatically fetches historical stock prices, company information, and financial statements from Yahoo Finance.
- 📰 **News Aggregation**: Incorporates financial news headlines from a pre-processed dataset on the Hugging Face Hub.
- ⚙️ **Efficient Data Processing**: Efficiently processes and merges data from various sources, using a caching mechanism to avoid redundant downloads.
- 🤖 **Specialized Prompt Creation**: Automatically formats the processed data into a structured prompt containing context, data, and response instructions for the model.
- 🧠 **Model Fine-tuning with GRPO**: Fine-tunes the `HuggingFaceTB/SmolLM2-1.7B-Instruct` model using the `trl` library and the `GRPOTrainer`. This approach uses reward functions to teach the model how to generate answers that are both well-formatted and accurate.
- ☁️ **Cloud Scalability**: Integrates with Modal to run the data collection process in a distributed cloud environment, significantly speeding it up.
- 🤗 **Hugging Face Integration**: Automatically uploads the final processed dataset and the fine-tuned model adapters to the Hugging Face Hub for easy access and sharing.

---

## ⚙️ Workflow Pipeline

The project's workflow is divided into four main stages:

### 1. Data Preparation
- Downloads the list of S&P 500 companies from Wikipedia.
- Collects news, financial statements, and historical price data for each company.
- Combines all information into a single dataset and uploads it to the Hugging Face Hub.

### 2. Prompt Engineering
- The raw data is converted into structured prompts to teach the model how to answer analytical questions for a specific day.

**Example Prompt Structure:**
```
You are a seasoned stock market analyst who is trying to predict whether the prices will go down or up over the day, {date}, for a specific stock, by offering a buy or sell rating.

[Company Name]
{company_name}

[Company Description]
{description}

[Price Movement]
It was {previous_close_price} on {previous_date}.
The price of the stock on {price_date} started at {open_price}.

[News since {news_start_date}]
{list_of_headlines}

[Financials]
{key_financials}

Your answer should look like the following
<think>reasoning...</think><answer>buy</answer>
or
<think>reasoning...</think><answer>sell</answer>
```

### 3. Model Training
- Loads the base model `HuggingFaceTB/SmolLM2-1.7B-Instruct`.
- Uses the `GRPOTrainer` from the `trl` library to train the model on the prompt-formatted dataset.
- The `GRPOTrainer` uses reward functions (`accuracy_reward` and `format_reward`) to "score" the model's outputs and guide it towards better performance.
- Training parameters like learning rate, batch size, and the number of epochs are configured to optimize performance.

### 4. Inference
- After training, the model can be used to predict intra-day price movement for a stock based on new data.

**Example Model Output:**

**User Input:**
```
Give me a buy or sell rating for Zoetis Inc. today.
```

**Model Output:**
```
<think>Based on the recent positive news, including an analyst upgrade and a new acquisition, there is strong positive sentiment. The stock opened higher than the previous close, suggesting initial bullish momentum. The financials like EPS are solid. Therefore, the stock is likely to continue its upward trend throughout the day.</think><answer>buy</answer>
```

---

## 🛠️ Technology Stack

- **Core**: Python  
- **Data Handling**: Pandas, yfinance, datasets  
- **LLM & Training**: transformers, peft, trl (GRPOTrainer)  
- **Cloud Computing**: Modal  
- **Platforms**: Hugging Face Hub, Wandb (Weights & Biases)

---

## 🚀 Usage Guide

### 1. Set Up Credentials
- Add your `HUGGING_FACE_HUB_TOKEN` and `WANDB_API_KEY` as secrets in your environment.
- Configure your Modal token.

### 2. Run the Scripts
- Execute the data preparation scripts to generate the cache files and the initial dataset on the Hugging Face Hub.
- Execute the training script to fine-tune the model.

💡 **Tip**: If you want to skip the time-consuming data preparation step, you can use the pre-processed dataset directly from here:  
`chuotchuilacduong/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2`

### 3. Access the Artifacts
- The final dataset and the trained model adapters will be available in your Hugging Face Hub repository.

---

## ⚠️ Disclaimer
This project is for educational and research purposes only.  
The data is collected from open sources and may contain inaccuracies.  
**This is not financial advice.**  
Do not use the dataset or models trained from this project to make actual investment decisions.
