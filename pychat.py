import sys
import json
import logging
import requests
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit, 
                            QSplitter, QMessageBox, QCheckBox, QTabWidget, QGridLayout,
                            QGroupBox, QFormLayout, QStackedWidget, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QFont, QIcon, QTextCursor

# Import OpenAI specific handler
from openai_handler import OpenAIRequestWorker, OpenAIModelsWorker
# Import Ollama specific handler
from ollama_handler import OllamaRequestWorker
# Import Anthropic specific handler
from anthropic_handler import AnthropicRequestWorker, AnthropicModelsWorker
# Import Preprompt manager
from preprompt_manager import PrepromptManager, CollapsiblePrepromptUI
# Import Enhanced Text Browser
from enhanced_chat_browser import EnhancedChatBrowser

# Setup logging
logging.basicConfig(filename='ai_chat_debug.log', level=logging.DEBUG,
                    format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

# Provider configurations
PROVIDERS = {
    "ollama": {
        "name": "Ollama",
        "api_url": "http://localhost:11434",  
        "auth_type": "none",
        "models_endpoint": "http://localhost:11434/api/tags",
        "models_field": "models",
        "model_name_field": "name",
        "streaming": True,
        "thinking_format": "<think>...</think>",
        "request_format": lambda model, prompt, stream: {
            "model": model,
            "prompt": prompt,
            "stream": stream
        },
        "response_field": "response",
        "streaming_field": "response"
    },
    "anthropic": {
        "name": "Claude (Anthropic)",
        "api_url": "https://api.anthropic.com/v1/messages",
        "auth_type": "api_key",
        "auth_header": "x-api-key",
        "anthropic_version": "2023-06-01",  # Add this line
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "streaming": True,
        "thinking_format": None,
        "request_format": lambda model, prompt, stream: {
            "model": model,
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream
        },
        "response_field": "content[0].text",
        "streaming_field": "delta.text"
    },
    "openai": {
        "name": "OpenAI",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "auth_type": "api_key",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "models_endpoint": "https://api.openai.com/v1/models",  
        "streaming": True,
        "thinking_format": None,
        "request_format": lambda model, prompt, stream: {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream
        },
        "response_field": "choices[0].message.content",
        "streaming_field": "choices[0].delta.content"
    }
}

import json
import logging
import requests
from PyQt5.QtCore import QThread, pyqtSignal

class RequestWorker(QThread):
    """Worker thread to handle API requests without freezing the UI"""
    finished = pyqtSignal(str, bool)  # Signal to emit response and success status
    chunk_received = pyqtSignal(str)  # Signal to emit chunks during streaming
    
    def __init__(self, provider_config, model, prompt, api_key=None, stream=True):
        super().__init__()
        self.provider_config = provider_config
        self.model = model
        self.prompt = prompt
        self.api_key = api_key
        self.stream = stream and provider_config["streaming"]
        self.in_think_section = False
        self.in_code_block = False
        self.code_block_buffer = ""
        
        # CRITICAL FIX: Add a buffer for text accumulation
        self.accumulated_text = ""
        self.buffer_size = 80  # Characters to accumulate before sending
        self.last_char = ""    # Track the last character
    
    def run(self):
        try:
            # Build request based on provider configuration
            data = self.provider_config["request_format"](self.model, self.prompt, self.stream)
            
            # Build headers based on authentication type
            headers = {"Content-Type": "application/json"}
            if self.provider_config["auth_type"] == "api_key":
                prefix = self.provider_config.get("auth_prefix", "")
                headers[self.provider_config["auth_header"]] = f"{prefix}{self.api_key}"
            
            # Add Anthropic version header if needed
            if "anthropic_version" in self.provider_config:
                headers["anthropic-version"] = self.provider_config["anthropic_version"]
            
            if self.stream:
                # Streaming request
                with requests.post(
                    self.provider_config["api_url"],
                    json=data,
                    headers=headers,
                    stream=True,
                    timeout=120
                ) as response:
                    if response.status_code == 200:
                        full_response = ""
                        for line in response.iter_lines():
                            if line:
                                try:
                                    # Different providers format streaming differently
                                    if self.provider_config.get("name") == "Ollama":
                                        # Ollama sends full JSON objects per line
                                        chunk = json.loads(line.decode('utf-8'))
                                        field_path = self.provider_config["streaming_field"]
                                        chunk_text = self._get_nested_value(chunk, field_path)
                                        
                                        if chunk_text:
                                            self._process_chunk(chunk_text, full_response)
                                    
                                    elif self.provider_config.get("name") == "Claude (Anthropic)":
                                        # Claude sends "event: " prefixed data
                                        line_text = line.decode('utf-8')
                                        if line_text.startswith("data: "):
                                            event_data = json.loads(line_text[6:])
                                            field_path = self.provider_config["streaming_field"]
                                            chunk_text = self._get_nested_value(event_data, field_path)
                                            
                                            if chunk_text:
                                                self._process_chunk(chunk_text, full_response)
                                    
                                    elif self.provider_config.get("name") == "OpenAI":
                                        # OpenAI sends "data: " prefixed chunks
                                        line_text = line.decode('utf-8')
                                        if line_text.startswith("data: ") and not line_text.startswith("data: [DONE]"):
                                            try:
                                                event_data = json.loads(line_text[6:])
                                                field_path = self.provider_config["streaming_field"]
                                                chunk_text = self._get_nested_value(event_data, field_path)
                                                
                                                if chunk_text:
                                                    self._process_chunk(chunk_text, full_response)
                                            
                                            except json.JSONDecodeError:
                                                # Sometimes OpenAI sends malformed JSON or [DONE]
                                                logging.error(f"Failed to decode OpenAI chunk: {line_text}")
                                    
                                    else:
                                        # Generic streaming fallback
                                        try:
                                            chunk = json.loads(line.decode('utf-8'))
                                            field_path = self.provider_config["streaming_field"]
                                            chunk_text = self._get_nested_value(chunk, field_path)
                                            
                                            if chunk_text:
                                                self._process_chunk(chunk_text, full_response)
                                        
                                        except Exception as e:
                                            logging.error(f"Failed to process generic stream chunk: {e}")
                                
                                except Exception as e:
                                    logging.error(f"Failed to process streaming chunk: {e}")
                        
                        # Final cleanup - send any remaining accumulated text
                        if self.accumulated_text:
                            full_response += self.accumulated_text
                            self.chunk_received.emit(self.accumulated_text)
                            self.accumulated_text = ""
                        
                        # Final cleanup - if we have a partial code block, send it
                        if self.code_block_buffer:
                            full_response += self.code_block_buffer
                            self.chunk_received.emit(self.code_block_buffer)
                        
                        # Send final complete response
                        self.finished.emit(full_response.strip(), True)
                    
                    else:
                        error_text = f"Error: API returned status code {response.status_code}"
                        try:
                            error_json = response.json()
                            if "error" in error_json:
                                error_text += f" - {error_json['error']}"
                        except:
                            pass
                        self.finished.emit(error_text, False)
            
            else:
                # Non-streaming request (fallback)
                response = requests.post(
                    self.provider_config["api_url"],
                    json=data,
                    headers=headers,
                    timeout=120
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    
                    # Extract the response text using the provider's response field path
                    field_path = self.provider_config["response_field"]
                    response_text = self._get_nested_value(response_json, field_path)
                    
                    if response_text:
                        self.finished.emit(response_text.strip(), True)
                    else:
                        self.finished.emit("Error: Could not extract response from model output", False)
                else:
                    error_text = f"Error: API returned status code {response.status_code}"
                    try:
                        error_json = response.json()
                        if "error" in error_json:
                            error_text += f" - {error_json['error']}"
                    except:
                        pass
                    self.finished.emit(error_text, False)
        
        except Exception as e:
            logging.error(f"Error in request: {str(e)}")
            self.finished.emit(f"Error: {str(e)}", False)
    
    def _process_chunk(self, chunk_text, full_response):
        """Process a chunk of text, handling special cases and buffering"""
        # Process any thinking sections
        if self.provider_config.get("thinking_format"):
            # Check if this is a Deepseek model
            is_deepseek = "deepseek" in self.model.lower()
            
            if is_deepseek:
                # Deepseek uses <thinking> tags
                if "<thinking>" in chunk_text:
                    self.in_think_section = True
                
                # Skip this chunk if we're in a thinking section
                if self.in_think_section:
                    if "</thinking>" in chunk_text:
                        self.in_think_section = False
                    return
            elif "<think>" in chunk_text:  # Ollama standard thinking format
                self.in_think_section = True
            
            # Skip this chunk if we're in a thinking section
            if self.in_think_section:
                if "</think>" in chunk_text:
                    self.in_think_section = False
                return
        
        # Special handling for code blocks
        if self.in_code_block:
            # We're inside a code block, keep accumulating
            self.code_block_buffer += chunk_text
            
            # Check if the code block is complete
            if "```" in chunk_text:
                # Code block is complete, emit it as one chunk
                self.in_code_block = False
                full_response += self.code_block_buffer
                
                # First send any accumulated text
                if self.accumulated_text:
                    self.chunk_received.emit(self.accumulated_text)
                    self.accumulated_text = ""
                
                # Then send the code block
                self.chunk_received.emit(self.code_block_buffer)
                self.code_block_buffer = ""
            return
        
        # Check if this chunk starts a code block
        if "```" in chunk_text and not chunk_text.count("```") % 2 == 0:
            # This starts a code block, begin accumulating
            self.in_code_block = True
            
            # First send any accumulated text
            if self.accumulated_text:
                full_response += self.accumulated_text
                self.chunk_received.emit(self.accumulated_text)
                self.accumulated_text = ""
            
            # Start accumulating the code block
            self.code_block_buffer = chunk_text
            return
        
        # CRITICAL FIX: For normal text, accumulate until we have a decent chunk size
        # or until we hit a natural break (newline or punctuation)
        self.accumulated_text += chunk_text
        self.last_char = chunk_text[-1] if chunk_text else self.last_char
        full_response += chunk_text
        
        # Send the accumulated text if:
        # 1. We hit a newline
        # 2. We accumulated enough characters
        # 3. We hit sentence-ending punctuation followed by space
        if ('\n' in self.accumulated_text or 
            (len(self.accumulated_text) >= self.buffer_size and 
             self.last_char in " .,;!?") or 
            re.search(r'[.!?]\s', self.accumulated_text)):
            
            self.chunk_received.emit(self.accumulated_text)
            self.accumulated_text = ""
    
    def _get_nested_value(self, obj, path):
        """Extract a value from a nested object using a dot-separated path"""
        if not obj:
            return None
            
        if "[" in path:  # Handle array access like choices[0].message.content
            parts = []
            current = ""
            for char in path:
                if char == "[":
                    if current:
                        parts.append(current)
                        current = ""
                    current = "["
                elif char == "]" and current.startswith("["):
                    current += "]"
                    parts.append(current)
                    current = ""
                else:
                    current += char
            if current:
                parts.append(current)
        else:
            parts = path.split(".")
        
        result = obj
        for part in parts:
            if part.startswith("[") and part.endswith("]"):
                # Handle array access
                try:
                    idx = int(part[1:-1])
                    if isinstance(result, list) and idx < len(result):
                        result = result[idx]
                    else:
                        return None
                except (ValueError, TypeError):
                    return None
            else:
                # Handle object property access
                if isinstance(result, dict) and part in result:
                    result = result[part]
                else:
                    return None
        return result


class OllamaModelsWorker(QThread):
    """Worker thread to load Ollama models"""
    finished = pyqtSignal(list, bool)
    
    def __init__(self, api_url):
        super().__init__()
        self.api_url = api_url
    
    def run(self):
        try:
            response = requests.get(self.api_url, timeout=10)
            
            if response.status_code == 200:
                response_json = response.json()
                
                # Extract model names
                models = []
                if "models" in response_json:
                    for model in response_json["models"]:
                        if "name" in model:
                            models.append(model["name"])
                
                self.finished.emit(models, True)
            else:
                self.finished.emit([], False)
        except Exception as e:
            logging.error(f"Error loading Ollama models: {str(e)}")
            self.finished.emit([], False)


class MultiProviderChat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Provider AI Chat")
        self.setMinimumSize(900, 700)
        
        # Initialize variables
        self.is_processing = False
        self.in_think_section = False
        self.response_placeholder_id = 0
        self.selected_provider = "ollama"  # Default provider
        self.selected_model = ""
        self.current_api_keys = {}
        self.last_used_models = {}  # Track the last used model for each provider
        
        # Load saved settings
        self.settings = QSettings("AI Chat App", "MultiProviderChat")
        self.load_settings()
        self.load_last_used_models()
        
        # Initialize PrepromptManager
        self.preprompt_manager = PrepromptManager(self, self.settings)
        
        # Setup the UI
        self.init_ui()
        
        # Log startup
        logging.debug("Application started. UI initialized.")
    
    def load_settings(self):
        """Load saved settings like API keys and URLs"""
        # Load API keys
        for provider_id in PROVIDERS:
            key = self.settings.value(f"api_keys/{provider_id}", "")
            if key:
                self.current_api_keys[provider_id] = key
        
        # Load custom API URLs
        for provider_id in PROVIDERS:
            url = self.settings.value(f"api_urls/{provider_id}", "")
            if url:
                PROVIDERS[provider_id]["api_url"] = url
        
        # Load last used provider
        last_provider = self.settings.value("last_provider", "ollama")
        if last_provider in PROVIDERS:
            self.selected_provider = last_provider
    
    def load_last_used_models(self):
        """Load last used model for each provider"""
        for provider_id in PROVIDERS:
            model = self.settings.value(f"last_used_models/{provider_id}", "")
            if model:
                self.last_used_models[provider_id] = model
    
    def save_settings(self):
        """Save current settings"""
        # Save API keys
        for provider_id, key in self.current_api_keys.items():
            self.settings.setValue(f"api_keys/{provider_id}", key)
        
        # Save custom API URLs
        for provider_id in PROVIDERS:
            if self.api_url_inputs.get(provider_id):
                url = self.api_url_inputs[provider_id].text()
                self.settings.setValue(f"api_urls/{provider_id}", url)
                PROVIDERS[provider_id]["api_url"] = url
        
        # Save last used provider
        self.settings.setValue("last_provider", self.selected_provider)
        
        # Save last used models
        for provider_id, model in self.last_used_models.items():
            self.settings.setValue(f"last_used_models/{provider_id}", model)
    
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create tab widget for different sections
        self.tabs = QTabWidget()
        
        # Create chat tab
        chat_tab = QWidget()
        chat_layout = QVBoxLayout()
        
        # ===== Provider selection section =====
        provider_group = QGroupBox("AI Provider")
        provider_layout = QHBoxLayout()
        
        # Provider dropdown
        provider_label = QLabel("Provider:")
        self.provider_dropdown = QComboBox()
        for provider_id, config in PROVIDERS.items():
            self.provider_dropdown.addItem(config["name"], provider_id)
        
        # Set the default provider
        for i in range(self.provider_dropdown.count()):
            if self.provider_dropdown.itemData(i) == self.selected_provider:
                self.provider_dropdown.setCurrentIndex(i)
                break
        
        self.provider_dropdown.currentIndexChanged.connect(self.on_provider_changed)
        
        # Server URL input (for Ollama)
        self.server_url_label = QLabel("Server URL:")
        self.server_url_input = QLineEdit(PROVIDERS["ollama"]["api_url"])
        self.server_url_input.setMinimumWidth(200)
        self.server_url_input.textChanged.connect(self.on_server_url_changed)
        
        # Model dropdown
        model_label = QLabel("Model:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.setMinimumWidth(150)
        
        # Refresh models button
        self.refresh_models_btn = QPushButton("Refresh Models")
        self.refresh_models_btn.clicked.connect(self.load_models)
        
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self.provider_dropdown)
        provider_layout.addWidget(self.server_url_label)
        provider_layout.addWidget(self.server_url_input, 1)
        provider_layout.addWidget(model_label)
        provider_layout.addWidget(self.model_dropdown)
        provider_layout.addWidget(self.refresh_models_btn)
        
        provider_group.setLayout(provider_layout)
        chat_layout.addWidget(provider_group)
        
        # ===== Options section =====
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout()
        
        # Streaming checkbox
        self.stream_checkbox = QCheckBox("Enable streaming")
        self.stream_checkbox.setChecked(True)
        self.stream_checkbox.setToolTip("Show responses in real-time as they are generated")
        
        # Show thinking checkbox
        self.show_thinking_checkbox = QCheckBox("Show thinking")
        self.show_thinking_checkbox.setChecked(False)
        self.show_thinking_checkbox.setToolTip("Show thinking sections in responses (if supported)")
        
        options_layout.addWidget(self.stream_checkbox)
        options_layout.addWidget(self.show_thinking_checkbox)
        options_layout.addStretch(1)
        
        options_group.setLayout(options_layout)
        chat_layout.addWidget(options_group)
        
        # Initialize CollapsiblePrepromptUI
        self.preprompt_ui = CollapsiblePrepromptUI(self, self.preprompt_manager)
        chat_layout.addWidget(self.preprompt_ui.get_preprompt_widget())
        
        # ===== Chat section =====
        # Create a splitter to allow resizing between chat history and input
        splitter = QSplitter(Qt.Vertical)
        
        # Chat history display
        self.chat_display = EnhancedChatBrowser()
        self.chat_display.setFont(QFont("Segoe UI", 10))
        # self.chat_display.setHtml("Welcome to Multi-Provider AI Chat!\n\nPlease select a provider and model to begin.\n")
        splitter.addWidget(self.chat_display)
        
        # User prompt input
        self.prompt_input = QTextEdit()
        self.prompt_input.setFont(QFont("Segoe UI", 10))
        self.prompt_input.setPlaceholderText("Type your message here...")
        self.prompt_input.setMinimumHeight(80)
        self.prompt_input.setMaximumHeight(150)
        splitter.addWidget(self.prompt_input)
        
        # Set initial sizes for the splitter
        splitter.setSizes([500, 100])
        
        chat_layout.addWidget(splitter, 1)  # Give the chat area most of the space
        
        # ===== Button section =====
        button_layout = QHBoxLayout()
        
        # Action buttons
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_prompt)
        self.send_btn.setMinimumHeight(40)
        
        self.clear_btn = QPushButton("Clear Chat")
        self.clear_btn.clicked.connect(self.clear_chat)
        self.clear_btn.setMinimumHeight(40)
        
        self.save_chat_btn = QPushButton("Save Chat")
        self.save_chat_btn.clicked.connect(self.save_chat)
        self.save_chat_btn.setMinimumHeight(40)
        
        button_layout.addWidget(self.send_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.save_chat_btn)
        
        chat_layout.addLayout(button_layout)
        
        chat_tab.setLayout(chat_layout)
        
        # ===== Settings Tab =====
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        
        # Provider settings (stacked widget)
        self.provider_settings = QStackedWidget()
        self.api_url_inputs = {}  # Store API URL inputs for each provider
        self.api_key_inputs = {}  # Store API key inputs for each provider
        
        # Create a settings page for each provider
        for provider_id, config in PROVIDERS.items():
            provider_page = QWidget()
            page_layout = QFormLayout()
            
            # API URL input
            if provider_id == "ollama":
                api_url_label = QLabel("Default API URL (can be changed in Chat tab):")
            else:
                api_url_label = QLabel(f"API URL:")
            
            api_url_input = QLineEdit(config["api_url"])
            self.api_url_inputs[provider_id] = api_url_input
            page_layout.addRow(api_url_label, api_url_input)
            
            # API key input for providers that need it
            if config["auth_type"] == "api_key":
                api_key_input = QLineEdit(self.current_api_keys.get(provider_id, ""))
                api_key_input.setEchoMode(QLineEdit.Password)
                self.api_key_inputs[provider_id] = api_key_input
                page_layout.addRow(f"API Key:", api_key_input)
            
            # Add save button
            save_btn = QPushButton("Save Settings")
            save_btn.clicked.connect(lambda checked, p=provider_id: self.save_provider_settings(p))
            page_layout.addRow("", save_btn)
            
            # Add provider documentation
            info_text = QLabel(f"<b>{config['name']} Information:</b><br>")
            if config["auth_type"] == "api_key":
                info_text.setText(info_text.text() + f"• Requires API key<br>")
            
            if provider_id == "anthropic":
                info_text.setText(info_text.text() + "• Visit https://console.anthropic.com/ to get API key<br>")
            elif provider_id == "openai":
                info_text.setText(info_text.text() + "• Visit https://platform.openai.com/ to get API key<br>")
            elif provider_id == "ollama":
                info_text.setText(info_text.text() + "• Local API, make sure Ollama is running<br>")
            
            page_layout.addRow(info_text)
            
            provider_page.setLayout(page_layout)
            self.provider_settings.addWidget(provider_page)
        
        # Provider selection for settings
        provider_select_layout = QHBoxLayout()
        provider_select_label = QLabel("Provider Settings:")
        self.settings_provider_dropdown = QComboBox()
        
        for provider_id, config in PROVIDERS.items():
            self.settings_provider_dropdown.addItem(config["name"], provider_id)
        
        self.settings_provider_dropdown.currentIndexChanged.connect(self.on_settings_provider_changed)
        
        provider_select_layout.addWidget(provider_select_label)
        provider_select_layout.addWidget(self.settings_provider_dropdown)
        provider_select_layout.addStretch(1)
        
        settings_layout.addLayout(provider_select_layout)
        settings_layout.addWidget(self.provider_settings)
        settings_layout.addStretch(1)
        
        settings_tab.setLayout(settings_layout)
        
        # Add tabs to main tab widget
        self.tabs.addTab(chat_tab, "Chat")
        self.tabs.addTab(settings_tab, "Settings")
        
        main_layout.addWidget(self.tabs)
        
        # Set the main layout to the widget
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Connect additional events
        self.prompt_input.installEventFilter(self)  # For Enter key detection
        self.prompt_input.installEventFilter(self)  # For Enter key detection
        self.model_dropdown.currentTextChanged.connect(self.update_thinking_checkbox_visibility)
        
        # Initialize provider and load models
        self.on_provider_changed()
    
    def eventFilter(self, source, event):
        # Allow sending with Ctrl+Enter
        if (event.type() == event.KeyPress and 
            source is self.prompt_input and 
            event.key() == Qt.Key_Return and 
            event.modifiers() & Qt.ControlModifier):
            self.send_prompt()
            return True
        return super().eventFilter(source, event)
    
    def on_provider_changed(self):
        """Handle provider selection change"""
        # Get selected provider ID
        index = self.provider_dropdown.currentIndex()
        self.selected_provider = self.provider_dropdown.itemData(index)
        
        # Show/hide server URL input based on provider
        if self.selected_provider == "ollama":
            self.server_url_label.setVisible(True)
            self.server_url_input.setVisible(True)
            self.server_url_input.setText(PROVIDERS["ollama"]["api_url"])
        else:
            self.server_url_label.setVisible(False)
            self.server_url_input.setVisible(False)
        
        # Show/hide thinking checkbox based on provider and model
        self.update_thinking_checkbox_visibility()
        
        # Update provider settings in settings tab
        for i in range(self.settings_provider_dropdown.count()):
            if self.settings_provider_dropdown.itemData(i) == self.selected_provider:
                self.settings_provider_dropdown.setCurrentIndex(i)
                break
        
        # Load models for the new provider
        self.load_models()
        
        # Save the setting
        self.save_settings()
        
    def update_thinking_checkbox_visibility(self):
        """Show thinking checkbox only for models that support it"""
        current_model = self.model_dropdown.currentText()
        
        # Show for Ollama's deepseek models, hide for others
        is_deepseek = self.selected_provider == "ollama" and "deepseek" in current_model.lower()
        self.show_thinking_checkbox.setVisible(is_deepseek)
    
    def on_server_url_changed(self):
        """Handle server URL change for Ollama"""
        if self.selected_provider == "ollama":
            base_url = self.server_url_input.text().strip()
            if base_url:
                # Ensure base URL doesn't end with a slash
                if base_url.endswith('/'):
                    base_url = base_url[:-1]
                    
                # Update provider config with proper endpoints
                # Note: The specialized OllamaRequestWorker will handle adding "/api/generate"
                PROVIDERS["ollama"]["api_url"] = base_url  # Store base URL without /api/generate
                PROVIDERS["ollama"]["models_endpoint"] = base_url + "/api/tags"
                
                # Also update the URL in settings tab
                if "ollama" in self.api_url_inputs:
                    self.api_url_inputs["ollama"].setText(base_url)
    
    def on_settings_provider_changed(self):
        """Handle provider selection change in the settings tab"""
        index = self.settings_provider_dropdown.currentIndex()
        provider_id = self.settings_provider_dropdown.itemData(index)
        
        # Change the stacked widget to show the selected provider's settings
        for i in range(self.provider_settings.count()):
            if i == index:
                self.provider_settings.setCurrentIndex(i)
                break
    
    def save_provider_settings(self, provider_id):
        """Save provider-specific settings"""
        # Save API URL
        if provider_id in self.api_url_inputs:
            url = self.api_url_inputs[provider_id].text().strip()
            if url:
                PROVIDERS[provider_id]["api_url"] = url
        
        # Save API key
        if provider_id in self.api_key_inputs:
            key = self.api_key_inputs[provider_id].text().strip()
            if key:
                self.current_api_keys[provider_id] = key
        
        # Save all settings
        self.save_settings()
        
        # Show confirmation
        QMessageBox.information(self, "Settings Saved", 
                               f"Settings for {PROVIDERS[provider_id]['name']} have been saved.")
    
    def load_models(self):
        """Load models for the current provider"""
        provider_config = PROVIDERS[self.selected_provider]
        
        # Clear existing models
        self.model_dropdown.clear()
        
        if self.selected_provider == "ollama":
            # For Ollama, we need to fetch models via API
            # Get the URL from the UI input
            api_url = self.server_url_input.text().strip()
            if not api_url:
                api_url = "http://localhost:11434"
                self.server_url_input.setText(api_url)
            
            # Update the provider config with this URL
            # Update the provider config with the correct endpoints
            PROVIDERS["ollama"]["api_url"] = api_url
            models_endpoint = api_url + "/api/tags"
            PROVIDERS["ollama"]["models_endpoint"] = models_endpoint
            
            self.append_to_chat(f"[SYSTEM] Loading models from {api_url}...")
            
            # Get models via API
            self.ollama_models_worker = OllamaModelsWorker(models_endpoint)
            self.ollama_models_worker.finished.connect(self.handle_ollama_models)
            self.ollama_models_worker.start()
        elif self.selected_provider == "openai":
            # For# For OpenAI, fetch available models via API
            if self.selected_provider not in self.current_api_keys:
                self.append_to_chat("[SYSTEM] API key required to load OpenAI models. Please set it in the Settings tab.")
                return
                
            self.append_to_chat("[SYSTEM] Loading models from OpenAI...")
            
            # Use the OpenAI models worker
            self.openai_models_worker = OpenAIModelsWorker(
                self.current_api_keys.get("openai", "")
            )
            self.openai_models_worker.finished.connect(self.handle_openai_models)
            self.openai_models_worker.start()
        elif self.selected_provider == "anthropic":
            # For Anthropic, fetch available models via our specialized worker
            if self.selected_provider not in self.current_api_keys:
                self.append_to_chat("[SYSTEM] API key required to load Anthropic models. Please set it in the Settings tab.")
                return
                
            self.append_to_chat("[SYSTEM] Loading models from Anthropic...")
            
            # Use the Anthropic models worker
            self.anthropic_models_worker = AnthropicModelsWorker(
                self.current_api_keys.get("anthropic", "")
            )
            self.anthropic_models_worker.finished.connect(self.handle_anthropic_models)
            self.anthropic_models_worker.start()
        else:
            # For other providers, models are predefined
            if "models" in provider_config:
                for model in provider_config["models"]:
                    self.model_dropdown.addItem(model)
                
                # Set last used model as default, or first model if not available
                if self.model_dropdown.count() > 0:
                    last_model = self.last_used_models.get(self.selected_provider, "")
                    found = False
                    if last_model:
                        for i in range(self.model_dropdown.count()):
                            if self.model_dropdown.itemText(i) == last_model:
                                self.model_dropdown.setCurrentIndex(i)
                                self.selected_model = last_model
                                found = True
                                break
                    if not found:
                        self.selected_model = self.model_dropdown.itemText(0)

    def handle_openai_models(self, models, success):
        """Handle loaded OpenAI models"""
        if success and models:
            for model in models:
                self.model_dropdown.addItem(model)
            
            self.append_to_chat(f"[SYSTEM] Loaded {len(models)} models from OpenAI.")
            
            # Set last used model as default, or first model if not available
            if self.model_dropdown.count() > 0:
                last_model = self.last_used_models.get(self.selected_provider, "")
                found = False
                if last_model:
                    for i in range(self.model_dropdown.count()):
                        if self.model_dropdown.itemText(i) == last_model:
                            self.model_dropdown.setCurrentIndex(i)
                            self.selected_model = last_model
                            found = True
                            break
                if not found:
                    self.selected_model = self.model_dropdown.itemText(0)
        else:
            self.append_to_chat("[SYSTEM] Error loading models from OpenAI. Check your API key and connection.")
            
    def handle_ollama_models(self, models, success):
        """Handle loaded Ollama models"""
        if success and models:
            for model in models:
                self.model_dropdown.addItem(model)
            
            self.append_to_chat(f"[SYSTEM] Loaded {len(models)} models from Ollama.")
            
            # Set last used model as default, or first model if not available
            if self.model_dropdown.count() > 0:
                last_model = self.last_used_models.get(self.selected_provider, "")
                found = False
                if last_model:
                    for i in range(self.model_dropdown.count()):
                        if self.model_dropdown.itemText(i) == last_model:
                            self.model_dropdown.setCurrentIndex(i)
                            self.selected_model = last_model
                            found = True
                            break
                if not found:
                    self.selected_model = self.model_dropdown.itemText(0)
        else:
            self.append_to_chat("[SYSTEM] Error loading models from Ollama. Make sure Ollama is running.")
    
    def handle_anthropic_models(self, models, success):
        """Handle loaded Anthropic models"""
        if success and models:
            for model in models:
                self.model_dropdown.addItem(model)
            
            self.append_to_chat(f"[SYSTEM] Loaded {len(models)} models from Anthropic.")
            
            # Set last used model as default, or first model if not available
            if self.model_dropdown.count() > 0:
                last_model = self.last_used_models.get(self.selected_provider, "")
                found = False
                if last_model:
                    for i in range(self.model_dropdown.count()):
                        if self.model_dropdown.itemText(i) == last_model:
                            self.model_dropdown.setCurrentIndex(i)
                            self.selected_model = last_model
                            found = True
                            break
                if not found:
                    self.selected_model = self.model_dropdown.itemText(0)
        else:
            self.append_to_chat("[SYSTEM] Error loading models from Anthropic. Check your API key and connection.")
    
    def send_prompt(self):
        if self.is_processing:
            self.append_to_chat("[SYSTEM] Already processing a request. Please wait.")
            return
        
        # Get user prompt
        base_prompt = self.prompt_input.toPlainText().strip()
        preprompt_text = self.preprompt_ui.get_current_preprompt_text()

        if not base_prompt:
            self.append_to_chat("[SYSTEM] Please enter a prompt.")
            return

        # Combine preprompt with user prompt if available
        if preprompt_text:
            prompt = preprompt_text + "\n\n" + base_prompt
            # Validate the combined prompt
            if not self.preprompt_ui.validate_prompt(prompt):
                self.append_to_chat("[SYSTEM] The combined preprompt and prompt contains syntax errors.")
                return
        else:
            prompt = base_prompt
        
        if not prompt:
            self.append_to_chat("[SYSTEM] Please enter a prompt.")
            return
        
        # Get selected provider and model
        provider_config = PROVIDERS[self.selected_provider]
        self.selected_model = self.model_dropdown.currentText()
        
        if not self.selected_model:
            self.append_to_chat("[SYSTEM] Please select a model first.")
            return
        
        # Check for API key if needed
        if provider_config["auth_type"] == "api_key" and self.selected_provider not in self.current_api_keys:
            self.append_to_chat(f"[SYSTEM] API key required for {provider_config['name']}. Please set it in the Settings tab.")
            self.tabs.setCurrentIndex(1)  # Switch to settings tab
            return
        
        # Show user message in chat
        self.append_to_chat(f"> {prompt}")
        
        # Clear prompt input
        self.prompt_input.clear()
        
        # Set processing state
        self.is_processing = True
        self.update_ui_state(enabled=False)
        
        # Create a placeholder for the streaming response
        self.response_placeholder_id = self.add_streaming_placeholder()
        
        # Check if streaming is enabled
        streaming_enabled = self.stream_checkbox.isChecked() and provider_config["streaming"]
        if streaming_enabled:
            self.current_streaming_id = self.chat_display.begin_streaming_response()
        
        # Use specialized handlers for different providers
        if self.selected_provider == "openai":
            self.worker = OpenAIRequestWorker(
                provider_config["api_url"], 
                self.selected_model, 
                prompt, 
                self.current_api_keys.get("openai", ""),
                stream=streaming_enabled
            )
        elif self.selected_provider == "ollama":
            # Use dedicated Ollama handler with base URL
            base_url = self.server_url_input.text().strip()
            if not base_url:
                base_url = "http://localhost:11434"  # Default
                
            self.worker = OllamaRequestWorker(
                base_url,
                self.selected_model, 
                prompt, 
                stream=streaming_enabled
            )
        else:
            # Use generic handler for other providers
            self.worker = RequestWorker(
                provider_config, 
                self.selected_model, 
                prompt, 
                api_key=self.current_api_keys.get(self.selected_provider),
                stream=streaming_enabled
            )
        
        # Common worker setup
        self.worker.finished.connect(self.handle_response)
        self.worker.chunk_received.connect(self.handle_chunk)
        self.worker.start()
        
    def handle_chunk(self, chunk_text):
        """Handle a chunk of streamed text from the AI provider"""
        if not chunk_text:  # Skip empty chunks
            return
                
        # Process thinking sections in streaming responses
        thinking_format = PROVIDERS[self.selected_provider]["thinking_format"]
        if thinking_format and not self.show_thinking_checkbox.isChecked():
            # Check if this is a Deepseek model
            is_deepseek = self.selected_provider == "ollama" and "deepseek" in self.selected_model.lower()
            
            if is_deepseek:
                # Deepseek uses <thinking> tags
                if "<thinking>" in chunk_text:
                    self.in_think_section = True
                
                # Skip this chunk if we're in a thinking section
                if self.in_think_section:
                    if "</thinking>" in chunk_text:
                        self.in_think_section = False
                    return
            elif "<think>" in chunk_text:  # Ollama standard thinking format
                self.in_think_section = True
            
            # Skip this chunk if we're in a thinking section
            if self.in_think_section:
                if "</think>" in chunk_text:
                    self.in_think_section = False
                return
        
        # If this is the first chunk, clear placeholder message
        current_text = self.chat_display.toPlainText()
        if "[SYSTEM] Processing request..." in current_text:
            # Clear the placeholder and start fresh
            self.append_to_chat("")  # Add a blank line to separate from user input
        
        # COMPLETELY NEW APPROACH: Use the streaming response session
        if hasattr(self, 'current_streaming_id') and self.current_streaming_id:
            self.chat_display.append_streaming_chunk(self.current_streaming_id, chunk_text)
        else:
            # Fallback to old method
            self.chat_display.insertHtml(chunk_text)
        
        # Process events to ensure UI updates
        QApplication.processEvents()
    
    def handle_response(self, response, success):
        """Handle the complete response from the AI provider"""
        # Reset any think section state for streaming
        self.in_think_section = False
        
        # For non-streaming or error cases
        if not success:
            self.append_to_chat(f"[SYSTEM] {response}")
            self.is_processing = False
            self.update_ui_state(enabled=True)
            return
        
        # Process thinking sections in full responses
        thinking_format = PROVIDERS[self.selected_provider]["thinking_format"]
        if thinking_format and not self.show_thinking_checkbox.isChecked() and success:
            is_deepseek = self.selected_provider == "ollama" and "deepseek" in self.selected_model.lower()
            
            if is_deepseek:
                # Remove Deepseek thinking sections
                think_start = response.find("<thinking>")
                while think_start > -1:
                    think_end = response.find("</thinking>", think_start)
                    if think_end > think_start:
                        response = response[:think_start] + response[think_end + 11:]  # 11 is length of "</thinking>"
                        think_start = response.find("<thinking>")
                    else:
                        break
            else:
                # Remove standard Ollama thinking sections
                think_start = response.find("<think>")
                while think_start > -1:
                    think_end = response.find("</think>", think_start)
                    if think_end > think_start:
                        response = response[:think_start] + response[think_end + 8:]  # 8 is length of "</think>"
                        think_start = response.find("<think>")
                    else:
                        break
        
        # For streaming responses, we might need to replace the content
        if success and self.stream_checkbox.isChecked() and hasattr(self, 'current_streaming_id'):
            self.chat_display.end_streaming_response(self.current_streaming_id)
            self.current_streaming_id = None
        elif success and not self.stream_checkbox.isChecked():
            # For non-streaming successful responses
            self.append_to_chat(response)
        
        # Save this as the last used model for this provider
        self.last_used_models[self.selected_provider] = self.selected_model
        
        # Reset processing state
        self.is_processing = False
        self.update_ui_state(enabled=True)
    
    def append_to_chat(self, text):
        """Add text to chat display and scroll to bottom"""
        self.chat_display.append(text)
        
        # Log user content
        if not text.startswith("[SYSTEM]") and not text.startswith(">"):
            logging.debug(f"Received response, length: {len(text)}")
    
    def add_streaming_placeholder(self):
        """Add a placeholder for streaming responses and return its position"""
        # Add placeholder text
        self.append_to_chat("[SYSTEM] Processing request...")
        return 0
    
    
    def on_model_changed(self, model_name):
        """Handle model selection change"""
        self.selected_model = model_name
        self.update_thinking_checkbox_visibility()
        
        
    def update_ui_state(self, enabled=True):
        """Update UI elements based on processing state"""
        self.send_btn.setEnabled(enabled)
        self.prompt_input.setEnabled(enabled)
        self.provider_dropdown.setEnabled(enabled)
        self.model_dropdown.setEnabled(enabled)
        self.refresh_models_btn.setEnabled(enabled)
        self.stream_checkbox.setEnabled(enabled)
        self.show_thinking_checkbox.setEnabled(enabled)
        
        # Server URL input (only for Ollama)
        if hasattr(self, 'server_url_input'):
            self.server_url_input.setEnabled(enabled)
    
    def clear_chat(self):
        """Clear the chat history"""
        self.chat_display.clear()
        provider_name = PROVIDERS[self.selected_provider]["name"]
        self.chat_display.setText(f"Chat cleared. Provider: {provider_name}, Model: {self.selected_model}\n\n")
        logging.debug("Chat cleared")
    
    def save_chat(self):
        """Save chat history to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chat History", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(self.chat_display.toPlainText())
                QMessageBox.information(self, "Chat Saved", f"Chat history saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not save chat: {str(e)}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Save settings before closing
        self.preprompt_manager.save_preprompts()
        self.save_settings()
        event.accept()


if __name__ == "__main__":
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look across platforms
    
    # Create and show the main window
    window = MultiProviderChat()
    window.show()
    
    # Run application
    sys.exit(app.exec_())