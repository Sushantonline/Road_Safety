# Road Accident Data Analysis System

An AI-powered system for analyzing Indian road accident data using natural language queries, automated visualization, and fine-tuned language models.

## Features

- 🤖 **Natural Language Queries**: Ask questions in plain English about road accident data
- 📊 **Automated Visualizations**: Get relevant plots and charts based on your queries
- 🧠 **Chain of Thought Reasoning**: Fine-tuned LLM with step-by-step reasoning
- 📈 **Interactive Dashboard**: Explore data through multiple views and interfaces
- 🔄 **Multiple Prompt Types**: Analysis, comparison, and chain-of-thought prompting
- 📱 **Responsive Design**: Works on desktop and mobile devices

### Directory Structure
    road_accident_analysis/
    │
    ├── data/
    │   ├── raw/
    │   │   ├── pdfs/
    │   │   └── extracted_text/
    │   ├── processed/
    │   │   ├── dataframes/
    │   │   └── cot_dataset/
    │   └── models/
    │
    ├── src/
    │   ├── __init__.py
    │   ├── data_processing/
    │   │   ├── __init__.py
    │   │   ├── pdf_parser.py
    │   │   ├── table_extractor.py
    │   │   └── text_processor.py
    │   ├── llm/
    │   │   ├── __init__.py
    │   │   ├── cot_generator.py
    │   │   ├── model_trainer.py
    │   │   └── query_processor.py
    │   ├── visualization/
    │   │   ├── __init__.py
    │   │   ├── plot_generator.py
    │   │   └── plot_recommender.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── config.py
    │       └── helpers.py
    │
    ├── streamlit_app/
    │   ├── __init__.py
    │   ├── app.py
    │   ├── components/
    │   │   ├── __init__.py
    │   │   ├── data_explorer.py
    │   │   ├── query_interface.py
    │   │   └── visualization_panel.py
    │   └── pages/
    │       ├── 01_Data_Overview.py
    │       ├── 02_Query_Interface.py
    │       ├── 03_Model_Training.py
    │       └── 04_Analytics_Dashboard.py
    │
    ├── config/
    │   ├── model_config.yaml
    │   └── data_config.yaml
    │
    ├── requirements.txt
    ├── setup.py
    └── README.md


## Installation

### 1. Clone the repository:
    git clone https://github.com/yourusername/road-accident-analysis.git
    cd road-accident-analysis


### 2. Create a virtual environment:
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate

### 3. Install dependencies:
    pip install -r requirements.txt

### 4. Install the package:
    pip install -e 

## Usage

### Running the Streamlit App
    streamlit run streamlit_app/app.py

### Training a Custom Model
    from src.llm.model_trainer import COTModelTrainer, DatasetGenerator
    from src.data_processing.text_processor import TextProcessor

Initialize components
text_processor = TextProcessor()
dataset_generator = DatasetGenerator(text_processor)
trainer = COTModelTrainer()

Create training data from your PDF texts
pdf_texts = ["your extracted text here..."]
training_examples = dataset_generator.create_training_dataset(pdf_texts)

Prepare and train
training_dataset = trainer.prepare_training_data(training_examples)
trainer.train_model(training_dataset)


### Using the Query Interface
    from src.llm.query_processor import QueryProcessor, ResponseGenerator

### Initialize
    processor = QueryProcessor()
    generator = ResponseGenerator(processor)
    
    Load your dataframes
    dataframes = {
    'yearly_stats': your_yearly_df,
    'state_wise': your_state_df,
    ... other dataframes
    }
    processor.load_dataframes(dataframes)

### Ask questions
    result = generator.generate_response(
    "What is the trend in road accidents over the past 5 years?",
    prompt_type="cot"
    )
    print(result['response'])


## Data Structure

The system expects dataframes with the following structure:

### Yearly Statistics
- `year`: Year (2018-2022)
- `accidents`: Total accidents
- `fatalities`: Total fatalities
- `injuries`: Total injuries
- `fatal_accidents`: Number of fatal accidents

### State-wise Data
- `state`: State name
- `accidents`: Number of accidents
- `fatalities`: Number of fatalities
- `population`: State population

### Road Type Data
- `road_type`: Type of road (National Highway, State Highway, etc.)
- `accidents`: Number of accidents
- `fatalities`: Number of fatalities
- `road_length_km`: Length in kilometers

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

