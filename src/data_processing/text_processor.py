import pandas as pd
import re
from typing import List, Dict, Tuple
from pathlib import Path
import json

class TextProcessor:
    def __init__(self):
        self.accident_patterns = {
            'statistics': r'(\d+[,.]?\d*)\s*(accidents?|fatalities|deaths?|injuries)',
            'percentages': r'(\d+\.?\d*)\s*percent',
            'years': r'(20\d{2})',
            'states': r'(Tamil Nadu|Maharashtra|Uttar Pradesh|Karnataka|Kerala|Gujarat|Rajasthan|Madhya Pradesh|Andhra Pradesh|Bihar|West Bengal|Telangana|Odisha|Haryana|Punjab|Assam|Chhattisgarh|Jharkhand|Uttarakhand|Himachal Pradesh|Tripura|Meghalaya|Manipur|Nagaland|Goa|Arunachal Pradesh|Mizoram|Sikkim)',
            'road_types': r'(National Highway|State Highway|district road|rural road|urban road)',
            'accident_types': r'(hit and run|head.?on collision|overturn|rear.?end|side collision)'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract relevant entities from text"""
        entities = {}
        for entity_type, pattern in self.accident_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = matches
        return entities
    
    def create_context_windows(self, text: str, window_size: int = 512, overlap: int = 100) -> List[str]:
        """Create overlapping context windows from text"""
        words = text.split()
        windows = []
        
        for i in range(0, len(words), window_size - overlap):
            window = ' '.join(words[i:i + window_size])
            if len(window.strip()) > 50:  # Minimum window size
                windows.append(window)
        
        return windows
    
    def generate_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """Generate question-answer pairs from text"""
        qa_pairs = []
        entities = self.extract_entities(text)
        
        # Generate statistics-based questions
        if entities.get('statistics'):
            for stat in entities['statistics']:
                if 'accident' in stat[1].lower():
                    question = f"How many accidents were reported?"
                    answer = f"According to the data, {stat[0]} {stat[1]} were reported."
                    qa_pairs.append({"question": question, "answer": answer, "context": text[:500]})
        
        # Generate comparative questions
        if entities.get('states') and len(entities['states']) > 1:
            states = entities['states'][:2]
            question = f"Compare accident statistics between {states[0]} and {states[1]}"
            answer = f"Based on the data, here's a comparison between {states[0]} and {states[1]}..."
            qa_pairs.append({"question": question, "answer": answer, "context": text[:500]})
        
        return qa_pairs

class COTDatasetGenerator:
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
    
    def generate_cot_examples(self, text: str) -> List[Dict[str, str]]:
        """Generate Chain of Thought examples"""
        cot_examples = []
        entities = self.text_processor.extract_entities(text)
        
        # Example 1: Multi-step reasoning
        if entities.get('statistics') and entities.get('years'):
            question = "What was the trend in road accidents over the years?"
            thinking = """
            Let me analyze this step by step:
            1. First, I need to identify the years mentioned in the data
            2. Then, I need to find the accident statistics for each year
            3. Compare the numbers to identify the trend
            4. Provide a conclusion about whether accidents are increasing or decreasing
            """
            answer = "Based on the data analysis, I can see the trend in road accidents..."
            
            cot_examples.append({
                "question": question,
                "chain_of_thought": thinking,
                "answer": answer,
                "context": text[:1000]
            })
        
        # Example 2: Cause analysis
        if entities.get('accident_types'):
            question = "What are the main causes of road accidents?"
            thinking = """
            To answer this question, I need to:
            1. Identify all accident types mentioned in the data
            2. Look for statistical information about each type
            3. Rank them by frequency or severity
            4. Provide insights about the most common causes
            """
            answer = "The main causes of road accidents based on the data are..."
            
            cot_examples.append({
                "question": question,
                "chain_of_thought": thinking,
                "answer": answer,
                "context": text[:1000]
            })
        
        return cot_examples
    
    def process_document(self, text: str) -> Dict[str, List]:
        """Process entire document to generate training data"""
        windows = self.text_processor.create_context_windows(text)
        
        all_qa_pairs = []
        all_cot_examples = []
        
        for window in windows:
            qa_pairs = self.text_processor.generate_qa_pairs(window)
            cot_examples = self.generate_cot_examples(window)
            
            all_qa_pairs.extend(qa_pairs)
            all_cot_examples.extend(cot_examples)
        
        return {
            "qa_pairs": all_qa_pairs,
            "cot_examples": all_cot_examples
        }
