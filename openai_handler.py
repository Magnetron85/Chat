import requests
import json
import logging
from PyQt5.QtCore import QThread, pyqtSignal

class OpenAIRequestWorker(QThread):
    """Worker thread specifically for OpenAI API requests"""
    finished = pyqtSignal(str, bool)  # Signal to emit response and success status
    chunk_received = pyqtSignal(str)  # Signal to emit chunks during streaming
    
    def __init__(self, api_url, model, prompt, api_key, stream=True, use_conversation=False):
        super().__init__()
        self.api_url = api_url
        self.model = model
        self.prompt = prompt
        self.api_key = api_key
        self.stream = stream
        self.use_conversation = use_conversation  # NEW parameter
    
    def run(self):
        try:
            # Build headers with authorization
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # NEW: Handle different formats based on conversation mode
            if self.use_conversation and isinstance(self.prompt, list):
                # Using conversation history - prompt is already a list of messages
                data = {
                    "model": self.model,
                    "messages": self.prompt,
                    "stream": self.stream
                }
            # Build request data for OpenAI
            if self.use_conversation and isinstance(self.prompt, list):
                # Using conversation history - prompt is already a list of messages
                data = {
                    "model": self.model,
                    "messages": self.prompt,
                    "stream": self.stream
                }
            else:
                # Standard prompt format
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": self.prompt}],
                    "stream": self.stream
                }
            
            if self.stream:
                # Streaming request for OpenAI
                full_response = ""
                try:
                    with requests.post(
                        self.api_url,
                        json=data,
                        headers=headers,
                        stream=True,
                        timeout=120
                    ) as response:
                        if response.status_code != 200:
                            self.finished.emit(f"Error: OpenAI returned status code {response.status_code}", False)
                            return
                        
                        # Process the streaming response
                        for line in response.iter_lines():
                            if not line:
                                continue
                                
                            line_text = line.decode('utf-8')
                            
                            # Skip the [DONE] message
                            if line_text == "data: [DONE]":
                                logging.debug("OpenAI stream complete")
                                continue
                                
                            # Process data chunks
                            if line_text.startswith("data: "):
                                try:
                                    json_str = line_text[6:]  # Remove "data: " prefix
                                    chunk_data = json.loads(json_str)
                                    
                                    # Check if we have content in this chunk
                                    if (chunk_data.get("choices") and 
                                        len(chunk_data["choices"]) > 0 and 
                                        chunk_data["choices"][0].get("delta") and 
                                        chunk_data["choices"][0]["delta"].get("content")):
                                        
                                        chunk_text = chunk_data["choices"][0]["delta"]["content"]
                                        full_response += chunk_text
                                        self.chunk_received.emit(chunk_text)
                                except json.JSONDecodeError:
                                    logging.error(f"Failed to decode OpenAI chunk: {line_text}")
                                except Exception as e:
                                    logging.error(f"Error processing OpenAI chunk: {str(e)}")
                        
                        # Emit the complete response
                        self.finished.emit(full_response.strip(), True)
                        
                except requests.RequestException as e:
                    self.finished.emit(f"Network error: {str(e)}", False)
                    
            else:
                # Non-streaming request
                try:
                    response = requests.post(
                        self.api_url,
                        json=data,
                        headers=headers,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        response_json = response.json()
                        if (response_json.get("choices") and 
                            len(response_json["choices"]) > 0 and 
                            response_json["choices"][0].get("message") and 
                            response_json["choices"][0]["message"].get("content")):
                            
                            content = response_json["choices"][0]["message"]["content"]
                            self.finished.emit(content, True)
                        else:
                            self.finished.emit("Error: Could not find content in OpenAI response", False)
                    else:
                        self.finished.emit(f"Error: OpenAI returned status code {response.status_code}", False)
                        
                except requests.RequestException as e:
                    self.finished.emit(f"Network error: {str(e)}", False)
                    
        except Exception as e:
            logging.error(f"Error in OpenAI request: {str(e)}")
            self.finished.emit(f"Error: {str(e)}", False)


class OpenAIModelsWorker(QThread):
    """Worker thread to load available OpenAI models"""
    finished = pyqtSignal(list, bool)
    
    def __init__(self, api_key, filter_chat_models=True):
        super().__init__()
        self.api_key = api_key
        self.filter_chat_models = filter_chat_models
        
    def run(self):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Call the models endpoint
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                models_data = response.json()
                
                all_models = []
                if "data" in models_data:
                    # Extract model IDs
                    all_models = [model["id"] for model in models_data["data"]]
                    
                    # Sort models alphabetically
                    all_models.sort()
                    
                    # Filter for chat models if requested
                    if self.filter_chat_models:
                        # Filter for gpt models that are designed for chat
                        chat_models = []
                        
                        # Prioritize these common models
                        priority_models = [
                            "gpt-4-turbo",
                            "gpt-4",
                            "gpt-4-32k",
                            "gpt-3.5-turbo",
                            "gpt-3.5-turbo-16k"
                        ]
                        
                        # Add priority models first if they exist
                        for model in priority_models:
                            matching = [m for m in all_models if model in m and not m.endswith("-vision")]
                            chat_models.extend(matching)
                        
                        # Add other chat models
                        for model in all_models:
                            # Add any other gpt models that look like chat models and aren't already added
                            if ("gpt" in model.lower() and 
                                ("turbo" in model.lower() or 
                                 "gpt-4" in model.lower() or 
                                 "gpt-3.5" in model.lower()) and
                                model not in chat_models and
                                not model.endswith("-vision")):
                                chat_models.append(model)
                        
                        all_models = chat_models
                
                self.finished.emit(all_models, True)
            else:
                error_msg = f"Error fetching models: Status code {response.status_code}"
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_msg += f" - {error_json['error']['message']}"
                except:
                    pass
                
                logging.error(error_msg)
                self.finished.emit([], False)
                
        except Exception as e:
            logging.error(f"Error loading OpenAI models: {str(e)}")
            self.finished.emit([], False)