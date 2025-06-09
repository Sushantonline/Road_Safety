import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd
from typing import Dict, List, Any
import json
from pathlib import Path
import logging

class COTModelTrainer:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens = {
            "pad_token": "<pad>",
            "additional_special_tokens": ["<question>", "<thinking>", "<answer>", "<context>"]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_training_data(self, cot_examples: List[Dict[str, str]]) -> Dataset:
        """Prepare Chain of Thought training data"""
        training_texts = []
        
        for example in cot_examples:
            # Format: <question>Q<thinking>CoT<answer>A
            formatted_text = (
                f"<question>{example['question']}"
                f"<thinking>{example['chain_of_thought']}"
                f"<answer>{example['answer']}"
            )
            training_texts.append(formatted_text)
        
        # Tokenize
        tokenized_data = self.tokenizer(
            training_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        dataset = Dataset.from_dict({
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"]
        })
        
        return dataset
    
    def train_model(self, 
                   training_dataset: Dataset,
                   output_dir: str = "data/models/fine_tuned",
                   num_epochs: int = 3,
                   batch_size: int = 4,
                   learning_rate: float = 5e-5):
        """Fine-tune the model on Chain of Thought data"""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            logging_dir=f"{output_dir}/logs",
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=training_dataset,
            tokenizer=self.tokenizer,
        )
        
        self.logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Model saved to {output_dir}")
        
        return trainer
    
    def generate_response(self, question: str, max_length: int = 200) -> str:
        """Generate response using the fine-tuned model"""
        prompt = f"<question>{question}<thinking>"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the thinking and answer parts
        if "<thinking>" in response and "<answer>" in response:
            thinking_start = response.find("<thinking>") + len("<thinking>")
            answer_start = response.find("<answer>")
            thinking = response[thinking_start:answer_start].strip()
            answer = response[answer_start + len("<answer>"):].strip()
            
            return f"**Thinking Process:**\n{thinking}\n\n**Answer:**\n{answer}"
        else:
            return response[len(prompt):].strip()

class DatasetGenerator:
    def __init__(self, text_processor):
        self.text_processor = text_processor
    
    def create_training_dataset(self, 
                              pdf_texts: List[str], 
                              output_path: str = "data/processed/cot_dataset/training_data.json"):
        """Create training dataset from PDF texts"""
        all_examples = []
        
        for text in pdf_texts:
            cot_generator = COTDatasetGenerator(self.text_processor)
            examples = cot_generator.process_document(text)
            
            all_examples.extend(examples['cot_examples'])
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_examples, f, indent=2)
        
        return all_examples
    
    def augment_dataset(self, base_examples: List[Dict], augmentation_factor: int = 3):
        """Augment dataset with variations"""
        augmented = []
        
        question_variations = [
            "What can you tell me about {}?",
            "Explain the situation regarding {}",
            "Provide an analysis of {}",
            "What are the key insights about {}?"
        ]
        
        for example in base_examples:
            augmented.append(example)  # Original
            
            # Create variations
            for i in range(augmentation_factor - 1):
                new_example = example.copy()
                # Simple augmentation - in practice, use more sophisticated methods
                new_example['question'] = example['question'].replace("What", "How")
                augmented.append(new_example)
        
        return augmented
