import requests
import json
import logging
from PyQt5.QtCore import QThread, pyqtSignal

class AnthropicRequestWorker(QThread):
    """Worker thread specifically for Anthropic (Claude) API requests"""
    finished = pyqtSignal(str, bool)  # Signal to emit response and success status
    chunk_received = pyqtSignal(str)  # Signal to emit chunks during streaming
    
    def __init__(self, api_url, model, prompt, api_key, stream=True, max_tokens=4000, use_conversation=False):
        super().__init__()
        self.api_url = api_url
        self.model = model
        self.prompt = prompt
        self.api_key = api_key
        self.stream = stream
        self.max_tokens = max_tokens
        self.use_conversation = use_conversation  # NEW parameter
    
    def run(self):
        try:
            # Build headers with authorization (Anthropic uses x-api-key)
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"  # API version header
            }
            
            # NEW: Handle different formats based on conversation mode
            # Build request data for Anthropic
            if self.use_conversation and isinstance(self.prompt, list):
                # Using conversation history - prompt is already a list of messages
                data = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": self.prompt,
                    "stream": self.stream
                }
            else:
                # Standard prompt format
                data = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": self.prompt}],
                    "stream": self.stream
                }
            
            if self.stream:
                # Streaming request for Anthropic
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
                            error_msg = f"Error: Anthropic returned status code {response.status_code}"
                            try:
                                error_json = response.json()
                                if "error" in error_json:
                                    error_msg += f" - {error_json['error']['message']}"
                            except:
                                pass
                            self.finished.emit(error_msg, False)
                            return
                        
                        # Process the streaming response
                        for line in response.iter_lines():
                            if not line:
                                continue
                                
                            line_text = line.decode('utf-8')
                            
                            # Skip if the line doesn't start with "data: "
                            if not line_text.startswith("data: "):
                                continue
                                
                            # Process data chunks
                            try:
                                json_str = line_text[6:]  # Remove "data: " prefix
                                
                                # Skip empty events or the [DONE] message
                                if json_str.strip() == "" or json_str.strip() == "[DONE]":
                                    continue
                                    
                                chunk_data = json.loads(json_str)
                                
                                # Check content type and get delta text
                                if (chunk_data.get("type") == "content_block_delta" and 
                                    chunk_data.get("delta") and 
                                    chunk_data["delta"].get("text")):
                                    
                                    chunk_text = chunk_data["delta"]["text"]
                                    full_response += chunk_text
                                    self.chunk_received.emit(chunk_text)
                                    
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode Anthropic chunk: {line_text}")
                            except Exception as e:
                                logging.error(f"Error processing Anthropic chunk: {str(e)}")
                        
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
                        
                        # The text is in content[0].text for Anthropic API v2
                        if (response_json.get("content") and 
                            len(response_json["content"]) > 0 and 
                            response_json["content"][0].get("type") == "text" and
                            response_json["content"][0].get("text")):
                            
                            content = response_json["content"][0]["text"]
                            self.finished.emit(content, True)
                        else:
                            self.finished.emit("Error: Could not find content in Anthropic response", False)
                    else:
                        error_msg = f"Error: Anthropic returned status code {response.status_code}"
                        try:
                            error_json = response.json()
                            if "error" in error_json:
                                error_msg += f" - {error_json['error']['message']}"
                        except:
                            pass
                        self.finished.emit(error_msg, False)
                        
                except requests.RequestException as e:
                    self.finished.emit(f"Network error: {str(e)}", False)
                    
        except Exception as e:
            logging.error(f"Error in Anthropic request: {str(e)}")
            self.finished.emit(f"Error: {str(e)}", False)


class AnthropicModelsWorker(QThread):
    """Worker thread to load available Anthropic models dynamically from the API"""
    finished = pyqtSignal(list, bool)
    
    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key
        
    def run(self):
        if not self.api_key:
            # If no API key is provided, return an empty list
            self.finished.emit([], False)
            return
            
        try:
            # API endpoint for listing models
            models_url = "https://api.anthropic.com/v1/models"
            
            # Headers for the request
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"  # Use the same version as in your other requests
            }
            
            # Make request to list models
            response = requests.get(
                models_url,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                models_data = response.json()
                
                # Extract model IDs from the response
                if "data" in models_data and isinstance(models_data["data"], list):
                    # Extract just the model IDs
                    models = [model["id"] for model in models_data["data"] if "id" in model]
                    self.finished.emit(models, True)
                else:
                    logging.error("Unexpected response format from Anthropic Models API")
                    self.finished.emit([], False)
            else:
                error_msg = f"Error: Anthropic returned status code {response.status_code}"
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_msg += f" - {error_json['error']['message']}"
                except:
                    pass
                logging.error(error_msg)
                self.finished.emit([], False)
                
        except requests.RequestException as e:
            logging.error(f"Network error when fetching Anthropic models: {str(e)}")
            self.finished.emit([], False)
        except Exception as e:
            logging.error(f"Error fetching Anthropic models: {str(e)}")
            self.finished.emit([], False)
