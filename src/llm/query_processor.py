import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class QueryProcessor:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.data_frames = {}
        
        # Add special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dataframes(self, df_dict: Dict[str, pd.DataFrame]):
        """Load preprocessed dataframes"""
        self.data_frames = df_dict
    
    def identify_intent(self, query: str) -> Dict[str, Any]:
        """Identify the intent and entities in the query"""
        intent_patterns = {
            'trend_analysis': r'(trend|over time|increase|decrease|change|growth)',
            'comparison': r'(compare|versus|vs|between|difference)',
            'statistics': r'(how many|total|number of|count|statistics)',
            'ranking': r'(top|highest|lowest|best|worst|rank)',
            'cause_analysis': r'(why|cause|reason|due to|because)',
            'visualization': r'(plot|chart|graph|show|visualize|display)'
        }
        
        entities = {
            'states': re.findall(r'(Tamil Nadu|Maharashtra|Uttar Pradesh|Karnataka|Kerala|Gujarat|Rajasthan|Madhya Pradesh|Andhra Pradesh|Bihar|West Bengal|Telangana|Odisha|Haryana|Punjab|Assam|Chhattisgarh|Jharkhand|Uttarakhand|Himachal Pradesh|Tripura|Meghalaya|Manipur|Nagaland|Goa|Arunachal Pradesh|Mizoram|Sikkim)', query, re.IGNORECASE),
            'years': re.findall(r'(20\d{2})', query),
            'road_types': re.findall(r'(national highway|state highway|district road|rural road|urban road)', query, re.IGNORECASE),
            'metrics': re.findall(r'(accident|fatality|injury|death)', query, re.IGNORECASE)
        }
        
        identified_intents = []
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                identified_intents.append(intent)
        
        return {
            'intents': identified_intents,
            'entities': entities,
            'needs_visualization': 'visualization' in identified_intents
        }
    
    def generate_sql_query(self, query: str, intent_info: Dict) -> str:
        """Generate SQL-like pandas query based on intent"""
        # This is a simplified version - in practice, you'd have more sophisticated mapping
        sql_templates = {
            'trend_analysis': "SELECT year, accidents, fatalities FROM yearly_stats ORDER BY year",
            'statistics': "SELECT COUNT(*) as total_accidents FROM accidents_data",
            'ranking': "SELECT state, accidents FROM state_wise ORDER BY accidents DESC LIMIT 10"
        }
        
        # Return appropriate template based on primary intent
        if intent_info['intents']:
            return sql_templates.get(intent_info['intents'][0], "SELECT * FROM main_data LIMIT 10")
        return "SELECT * FROM main_data LIMIT 10"
    
    def execute_data_query(self, intent_info: Dict) -> pd.DataFrame:
        """Execute query on dataframes based on intent"""
        if not self.data_frames:
            return pd.DataFrame()
        
        # Get the most relevant dataframe based on intent
        if 'trend_analysis' in intent_info['intents']:
            if 'yearly_stats' in self.data_frames:
                return self.data_frames['yearly_stats']
        elif 'comparison' in intent_info['intents'] and intent_info['entities']['states']:
            if 'state_wise' in self.data_frames:
                states = intent_info['entities']['states']
                df = self.data_frames['state_wise']
                if 'state' in df.columns:
                    return df[df['state'].isin(states)]
        elif 'ranking' in intent_info['intents']:
            if 'state_wise' in self.data_frames:
                df = self.data_frames['state_wise']
                if 'accidents' in df.columns:
                    return df.nlargest(10, 'accidents')
        
        # Default: return first available dataframe
        return list(self.data_frames.values())[0] if self.data_frames else pd.DataFrame()

class PromptTemplates:
    @staticmethod
    def get_analysis_prompt(query: str, data_summary: str) -> str:
        return f"""
        You are an expert analyst of road accident data in India. Based on the following data and user query, provide a comprehensive analysis.

        User Query: {query}

        Data Summary:
        {data_summary}

        Please provide:
        1. Direct answer to the question
        2. Key insights from the data
        3. Trends or patterns you observe
        4. Recommendations if applicable

        Analysis:
        """
    
    @staticmethod
    def get_cot_prompt(query: str, data_summary: str) -> str:
        return f"""
        You are analyzing road accident data. Think step by step to answer the user's question.

        User Query: {query}

        Available Data:
        {data_summary}

        Let me think through this step by step:

        Step 1: Understanding the question
        [Analyze what the user is asking for]

        Step 2: Identifying relevant data
        [Determine which data points are needed]

        Step 3: Analysis
        [Perform the analysis]

        Step 4: Conclusion
        [Provide the final answer]

        Chain of Thought:
        """
    
    @staticmethod
    def get_comparison_prompt(query: str, data_summary: str) -> str:
        return f"""
        You are comparing road accident statistics. Provide a detailed comparison based on the data.

        Query: {query}

        Data:
        {data_summary}

        Comparison Analysis:
        1. Key differences:
        2. Similarities:
        3. Trends:
        4. Insights:

        Response:
        """

class ResponseGenerator:
    def __init__(self, query_processor: QueryProcessor):
        self.query_processor = query_processor
        self.templates = PromptTemplates()
        
    def generate_response(self, query: str, prompt_type: str = "analysis") -> Dict[str, Any]:
        """Generate response to user query"""
        # Analyze intent
        intent_info = self.query_processor.identify_intent(query)
        
        # Get relevant data
        data_result = self.query_processor.execute_data_query(intent_info)
        data_summary = self._create_data_summary(data_result)
        
        # Choose appropriate prompt
        if prompt_type == "cot":
            prompt = self.templates.get_cot_prompt(query, data_summary)
        elif prompt_type == "comparison":
            prompt = self.templates.get_comparison_prompt(query, data_summary)
        else:
            prompt = self.templates.get_analysis_prompt(query, data_summary)
        
        # Generate response (simplified - in practice, you'd use the fine-tuned model)
        response = self._generate_with_model(prompt)
        
        return {
            'response': response,
            'data_used': data_result,
            'intent_info': intent_info,
            'needs_visualization': intent_info['needs_visualization']
        }
    
    def _create_data_summary(self, df: pd.DataFrame) -> str:
        """Create a summary of the dataframe for the prompt"""
        if df.empty:
            return "No relevant data found."
        
        summary = f"Data shape: {df.shape}\n"
        summary += f"Columns: {', '.join(df.columns)}\n"
        
        if len(df) > 0:
            summary += f"Sample data:\n{df.head().to_string()}\n"
            
            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary += f"\nBasic statistics:\n{df[numeric_cols].describe().to_string()}"
        
        return summary
    
    def _generate_with_model(self, prompt: str) -> str:
        """Generate response using the language model"""
        try:
            inputs = self.query_processor.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.query_processor.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.query_processor.tokenizer.eos_token_id
                )
            
            response = self.query_processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            response = response[len(prompt):].strip()
            
            return response if response else "I need more specific information to provide a detailed analysis."
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
