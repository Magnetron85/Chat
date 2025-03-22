import requests
import json
import logging
from PyQt5.QtCore import QThread, pyqtSignal

class OllamaRequestWorker(QThread):
    """Worker thread specifically for Ollama API requests"""
    finished = pyqtSignal(str, bool)  # Signal to emit response and success status
    chunk_received = pyqtSignal(str)  # Signal to emit chunks during streaming
    
    def __init__(self, base_url, model, prompt, stream=True, use_conversation=False):
        super().__init__()
        # Fix Ollama API URL structure
        if base_url.endswith('/api/generate'):
            self.api_url = base_url  # Already correct
        elif base_url.endswith('/'):
            self.api_url = f"{base_url.rstrip('/')}api/generate"
        else:
            self.api_url = f"{base_url}/api/generate"
            
        # Log the API URL to help with debugging
        logging.debug(f"Ollama API URL: {self.api_url}")
        
        self.model = model
        self.prompt = prompt
        self.stream = stream
        self.use_conversation = use_conversation  # NEW parameter (for interface consistency)  # This is mainly for interface consistency
                                                 # Ollama doesn't have a native conversation API,
                                                 # so we handle it in the main app by constructing the prompt
                                                
    
    def run(self):
        try:
            # Build request data for Ollama
            data = {
                "model": self.model,
                "prompt": self.prompt,
                "stream": self.stream,
                "options": {}  # Can add Ollama-specific options here
            }
            
            # Build headers
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.stream:
                # Streaming request for Ollama
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
                            error_msg = f"Error: Ollama returned status code {response.status_code}"
                            try:
                                error_json = response.json()
                                if "error" in error_json:
                                    error_msg += f" - {error_json['error']}"
                            except:
                                pass
                            self.finished.emit(error_msg, False)
                            return
                        
                        # Process the streaming response
                        for line in response.iter_lines():
                            if not line:
                                continue
                                
                            try:
                                chunk_data = json.loads(line.decode('utf-8'))
                                
                                # Check if we have content in this chunk
                                if "response" in chunk_data:
                                    chunk_text = chunk_data["response"]
                                    full_response += chunk_text
                                    self.chunk_received.emit(chunk_text)
                                    
                                # Check if this is the done message
                                if chunk_data.get("done", False):
                                    logging.debug("Ollama stream complete")
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode Ollama chunk: {line}")
                            except Exception as e:
                                logging.error(f"Error processing Ollama chunk: {str(e)}")
                        
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
                        if "response" in response_json:
                            content = response_json["response"]
                            self.finished.emit(content, True)
                        else:
                            self.finished.emit("Error: Could not find content in Ollama response", False)
                    else:
                        error_msg = f"Error: Ollama returned status code {response.status_code}"
                        try:
                            error_json = response.json()
                            if "error" in error_json:
                                error_msg += f" - {error_json['error']}"
                        except:
                            pass
                        self.finished.emit(error_msg, False)
                        
                except requests.RequestException as e:
                    self.finished.emit(f"Network error: {str(e)}", False)
                    
        except Exception as e:
            logging.error(f"Error in Ollama request: {str(e)}")
            self.finished.emit(f"Error: {str(e)}", False)