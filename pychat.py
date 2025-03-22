import sys
import json
import logging
import requests
import os
import re
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit, 
                            QSplitter, QMessageBox, QCheckBox, QTabWidget, QGridLayout,
                            QGroupBox, QFormLayout, QStackedWidget, QFileDialog, QAction, QInputDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer
from PyQt5.QtGui import QFont, QIcon, QTextCursor, QTextCharFormat, QColor

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
from chat_db_manager import ChatDatabaseManager
from thread_ui_components import ThreadListWidget, SearchResultsWidget, AIToChatPanel, ThreadCreationDialog
from ai2ai_conversation_worker import AI2AIConversationWorker
import sqlite3

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
        "thinking_format": "deepseek",
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
        
        # NEW: Add conversation history tracking
        self.conversation_history = []  # Store the conversation messages
        self.memory_enabled = True  # Default to enabled
        
        # NEW: Initialize database manager
        self.db_manager = ChatDatabaseManager()
        self.current_thread_id = None
        
        # Load saved settings
        self.settings = QSettings("AI Chat App", "MultiProviderChat")
        self.load_settings()
        self.load_last_used_models()
        
        # Initialize PrepromptManager
        self.preprompt_manager = PrepromptManager(self, self.settings)
        
        # Setup the UI
        self.init_ui()
        
        # Create menu bar
        self.create_menu_bar()

        # Load the last thread if available
        self.load_last_thread()
        
        # Log startup
        logging.debug("Application started. UI initialized.")
        
        self.cached_models = {
            "ollama": [],
            "openai": [],
            "anthropic": []
        }
    
    def load_settings(self):
        """Load saved settings like API keys and URLs"""
        # Existing code
        for provider_id in PROVIDERS:
            key = self.settings.value(f"api_keys/{provider_id}", "")
            if key:
                self.current_api_keys[provider_id] = key
        
        for provider_id in PROVIDERS:
            url = self.settings.value(f"api_urls/{provider_id}", "")
            if url:
                PROVIDERS[provider_id]["api_url"] = url
        
        last_provider = self.settings.value("last_provider", "ollama")
        if last_provider in PROVIDERS:
            self.selected_provider = last_provider
        
        # Load memory enabled setting
        self.memory_enabled = self.settings.value("memory_enabled", True, type=bool)
        
        # NEW: Load last thread ID
        self.last_thread_id = self.settings.value("last_thread_id", None)
    
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
            
    def load_last_thread(self):
        """Load the last used thread if available"""
        if hasattr(self, 'last_thread_id') and self.last_thread_id:
            try:
                thread_id = int(self.last_thread_id)
                self.load_thread(thread_id)
            except:
                pass
            
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create tab widget for different sections
        self.tabs = QTabWidget()
        
        # Create chat tab
        chat_tab = QWidget()
        chat_layout = QVBoxLayout()
        
        # NEW: Create a horizontal splitter for thread list and chat area
        thread_chat_splitter = QSplitter(Qt.Horizontal)
        
        # NEW: Thread list panel
        thread_list_panel = QWidget()
        thread_list_layout = QVBoxLayout()
        
        # NEW: Create thread list widget
        self.thread_list_widget = ThreadListWidget(self.db_manager)
        self.thread_list_widget.thread_selected.connect(self.load_thread)
        thread_list_layout.addWidget(self.thread_list_widget)
        
        thread_list_panel.setLayout(thread_list_layout)
        thread_list_panel.setMinimumWidth(250)
        thread_list_panel.setMaximumWidth(400)
        
        # Add thread list to splitter
        thread_chat_splitter.addWidget(thread_list_panel)
        
        # Chat area
        chat_area = QWidget()
        chat_area_layout = QVBoxLayout()
        
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
        chat_area_layout.addWidget(provider_group)
        
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

        # Memory checkbox
        self.memory_checkbox = QCheckBox("Enable conversation memory")
        self.memory_checkbox.setChecked(self.memory_enabled)
        self.memory_checkbox.setToolTip("Maintain context between messages")
        self.memory_checkbox.stateChanged.connect(self.on_memory_toggled)

        options_layout.addWidget(self.stream_checkbox)
        options_layout.addWidget(self.show_thinking_checkbox)
        options_layout.addWidget(self.memory_checkbox)
        options_layout.addStretch(1)
        
        options_group.setLayout(options_layout)
        chat_area_layout.addWidget(options_group)
        
        # Initialize CollapsiblePrepromptUI
        self.preprompt_ui = CollapsiblePrepromptUI(self, self.preprompt_manager)
        chat_area_layout.addWidget(self.preprompt_ui.get_preprompt_widget())
        
        # ===== Chat section =====
        # Create a splitter to allow resizing between chat history and input
        splitter = QSplitter(Qt.Vertical)
        
        # Chat history display
        self.chat_display = EnhancedChatBrowser()
        self.chat_display.setFont(QFont("Segoe UI", 10))
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
        
        chat_area_layout.addWidget(splitter, 1)  # Give the chat area most of the space
        
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
        
        chat_area_layout.addLayout(button_layout)
        
        chat_area.setLayout(chat_area_layout)
        
        # Add chat area to splitter
        thread_chat_splitter.addWidget(chat_area)
        
        # Set the main layout of the chat tab
        chat_layout.addWidget(thread_chat_splitter)
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
        
        # NEW: Add Search Tab
        search_tab = QWidget()
        search_layout = QVBoxLayout()
        
        self.search_widget = SearchResultsWidget(self.db_manager)
        self.search_widget.result_selected.connect(self.handle_search_result)
        
        search_layout.addWidget(self.search_widget)
        search_tab.setLayout(search_layout)
        
        # NEW: Add AI-to-AI Tab
        ai_to_ai_tab = QWidget()
        ai_to_ai_layout = QVBoxLayout()
        
        self.ai_to_ai_panel = AIToChatPanel(self.db_manager, PROVIDERS)
        
        ai_to_ai_layout.addWidget(self.ai_to_ai_panel)
        ai_to_ai_tab.setLayout(ai_to_ai_layout)
        
        # Add tabs to main tab widget
        self.tabs.addTab(chat_tab, "Chat")
        # self.tabs.addTab(search_tab, "Search")
        self.tabs.addTab(settings_tab, "Settings")
        self.tabs.addTab(ai_to_ai_tab, "AI-to-AI")
        
        main_layout.addWidget(self.tabs)
        
        # Set the main layout to the widget
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Connect additional events
        self.prompt_input.installEventFilter(self)  # For Enter key detection
        self.model_dropdown.currentTextChanged.connect(self.update_thinking_checkbox_visibility)
        
        # Initialize provider and load models
        self.on_provider_changed()
        
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # New thread action
        new_thread_action = QAction("New Thread", self)
        new_thread_action.setShortcut("Ctrl+N")
        new_thread_action.triggered.connect(self.create_new_thread)
        file_menu.addAction(new_thread_action)
        
        # Advanced new thread action
        advanced_new_thread_action = QAction("Advanced New Thread...", self)
        advanced_new_thread_action.triggered.connect(self.create_advanced_new_thread)
        file_menu.addAction(advanced_new_thread_action)
        
        file_menu.addSeparator()
        
        # Import thread action
        import_thread_action = QAction("Import Thread...", self)
        import_thread_action.triggered.connect(self.import_thread)
        file_menu.addAction(import_thread_action)
        
        # Export thread action
        export_thread_action = QAction("Export Thread...", self)
        export_thread_action.triggered.connect(self.export_thread)
        file_menu.addAction(export_thread_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Thread menu
        thread_menu = menubar.addMenu("&Thread")
        
        # Rename thread action
        rename_thread_action = QAction("Rename Thread...", self)
        rename_thread_action.triggered.connect(self.rename_current_thread)
        thread_menu.addAction(rename_thread_action)
        
        # Archive thread action
        self.archive_thread_action = QAction("Archive Thread", self)
        self.archive_thread_action.triggered.connect(lambda: self.toggle_thread_archive(True))
        thread_menu.addAction(self.archive_thread_action)
        
        # Unarchive thread action
        self.unarchive_thread_action = QAction("Unarchive Thread", self)
        self.unarchive_thread_action.triggered.connect(lambda: self.toggle_thread_archive(False))
        self.unarchive_thread_action.setVisible(False)  # Hide initially
        thread_menu.addAction(self.unarchive_thread_action)
        
        thread_menu.addSeparator()
        
        # Delete thread action
        delete_thread_action = QAction("Delete Thread", self)
        delete_thread_action.triggered.connect(self.delete_current_thread)
        thread_menu.addAction(delete_thread_action)
    
    def on_memory_toggled(self, state):
        """Handle memory checkbox toggle"""
        self.memory_enabled = state == Qt.Checked
        self.settings.setValue("memory_enabled", self.memory_enabled)
        logging.debug(f"Conversation memory {'enabled' if self.memory_enabled else 'disabled'}")
        
    def load_thread(self, thread_id):
        """Load a chat thread and display its messages"""
        if self.is_processing:
            self.append_to_chat("[SYSTEM] Already processing a request. Please wait.")
            return
            
        thread = self.db_manager.get_thread(thread_id)
        if not thread:
            self.append_to_chat("[SYSTEM] Error loading thread.")
            return
        
        # Get all messages for this thread
        messages = self.db_manager.get_messages(thread_id)
        logging.debug(f"Loading thread {thread_id} with {len(messages)} messages")
        
        # Clear current chat display
        self.chat_display.clear()
        
        # Set window title to include thread title
        self.setWindowTitle(f"Multi-Provider AI Chat - {thread['title']}")
        
        # Display thread info
        provider = thread.get("provider", "")
        model = thread.get("model", "")
        created_at = thread.get("created_at", "")
        
        # Format creation date
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(created_at)
            formatted_time = dt.strftime("%m/%d/%Y %I:%M %p")
        except:
            formatted_time = created_at
        

        self.append_to_chat(f"[SYSTEM] Thread: {thread['title']}")
        self.append_to_chat(f"[SYSTEM] Created: {formatted_time}")
        
        self.append_to_chat("")  # Empty line for spacing
        
        # CRITICAL FIX: Display messages with proper role formatting
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Log message details for debugging
            logging.debug(f"Displaying message: role={role}, content_length={len(content)}, content_start={content[:30] if content else 'None'}...")
            
            if role == "user":
                # User messages prefixed with ">"
                self.format_message_with_code_blocks(content, True)
            elif role == "assistant":
                # Use message-specific provider and model if available,
                # falling back to the thread’s values if not.
                self.format_message_with_code_blocks(content, False)
                msg_provider = message.get("provider", self.selected_provider)
                msg_model = message.get("model", self.selected_model)
                self.append_to_chat(f"[SYSTEM] Response from: {msg_provider.capitalize()} - {msg_model}")
            elif role == "system":
                self.append_to_chat(f"[SYSTEM] {content}")
            else:
                logging.warning(f"Unknown message role: {role}")
                self.append_to_chat(f"[{role}] {content}")

        
        # Update current thread ID
        self.current_thread_id = thread_id
        
        # If memory is enabled, load conversation history from the database
        if self.memory_enabled:
            self.conversation_history = self.db_manager.get_conversation_history(thread_id)
            logging.debug(f"Loaded conversation history with {len(self.conversation_history)} messages")
        
        # Update UI based on thread's provider and model
        if provider and model:
            # Find provider in dropdown
            for i in range(self.provider_dropdown.count()):
                if self.provider_dropdown.itemData(i) == provider:
                    self.provider_dropdown.setCurrentIndex(i)
                    break
            
            # Wait for models to load, then select the correct model
            self.selected_model = model
            self.selected_provider = provider
            
            # Add model attribution
            # self.append_to_chat(f"[SYSTEM] Response from: {self.selected_provider.capitalize()} - {self.selected_model}")
            
            # Set a timer to try setting the correct model after models are loaded
            QTimer.singleShot(500, lambda: self.select_model_in_dropdown(model))
        
        # Update archive/unarchive UI if available
        if hasattr(self, 'update_thread_archive_actions'):
            self.update_thread_archive_actions(thread_id)
        
        # Set focus to the input field
        self.prompt_input.setFocus()

    def select_model_in_dropdown(self, model_name):
        """Helper to select a model in the dropdown after it's loaded"""
        for i in range(self.model_dropdown.count()):
            if self.model_dropdown.itemText(i) == model_name:
                self.model_dropdown.setCurrentIndex(i)
                break

    def handle_search_result(self, thread_id, message_id=None):
        """Handle a search result selection by loading the thread and scrolling to the message"""
        self.tabs.setCurrentIndex(0)  # Switch to chat tab
        self.load_thread(thread_id)
        
        if message_id:
            # Scroll to the specific message (not implemented yet)
            # This would require adding message IDs to the chat display or other mechanism
            pass

    # Update the send_prompt method to handle thread storage:
    def send_prompt(self):
        """Send a prompt to the AI provider and handle the response"""
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
            # Update the formatting to clearly separate preprompt from user instruction
            prompt = f"System: {preprompt_text}\n\nUser: {base_prompt}"
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
            self.tabs.setCurrentIndex(2)  # Switch to settings tab
            return
        
        # CRITICAL FIX: Check if we have a current thread, create one if not
        if not self.current_thread_id:
            # Create a new thread with a default title based on the first prompt
            title = base_prompt[:30] + ("..." if len(base_prompt) > 30 else "")
            self.current_thread_id = self.db_manager.create_thread(
                title,
                provider=self.selected_provider,
                model=self.selected_model,
                preprompt=preprompt_text
            )
            
            # Update thread list
            if hasattr(self, 'thread_list_widget') and self.db_manager:
                self.db_manager._emit_thread_list_changed()
        
        # CRITICAL FIX: Save the user message to the database with role="user" BEFORE processing
        if self.current_thread_id:
            logging.debug(f"Saving user prompt to database: thread_id={self.current_thread_id}, content={base_prompt[:30]}...")
            success = self.db_manager.add_message(
                self.current_thread_id,
                "user",  # Explicitly set the role to "user"
                prompt,  # Store the prompt 
                provider=self.selected_provider,
                model=self.selected_model
            )
            if not success:
                logging.error("Failed to save user message to database")
        
        # Show user message in chat display
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
        
        # Add current message to conversation history
        user_message = {"role": "user", "content": prompt}
        
        # Handle conversation history based on memory toggle
        final_prompt = prompt
        if self.memory_enabled and hasattr(self, 'conversation_history') and self.conversation_history:
            # Different handling based on provider
            if self.selected_provider == "openai" or self.selected_provider == "anthropic":
                # These providers have native conversation handling - pass the full history
                # We don't modify the prompt but instead will modify the request format
                pass
            else:
                # For providers without native conversation support (e.g., Ollama),
                # we need to construct a prompt that includes the conversation history
                history_text = ""
                for message in self.conversation_history:
                    if message["role"] == "user":
                        history_text += f"User: {message['content']}\n\n"
                    else:
                        history_text += f"Assistant: {message['content']}\n\n"
                
                # Append the current prompt
                final_prompt = f"{history_text}User: {prompt}\n\nAssistant:"
        
        # Use specialized handlers for different providers
        if self.selected_provider == "openai":
            # Handle OpenAI conversation history
            if self.memory_enabled and self.conversation_history:
                # Include conversation history in messages
                messages = self.conversation_history.copy()
                messages.append(user_message)
                self.worker = OpenAIRequestWorker(
                    provider_config["api_url"], 
                    self.selected_model, 
                    messages, 
                    self.current_api_keys.get("openai", ""),
                    stream=streaming_enabled,
                    use_conversation=True  # Flag to indicate we're using conversation history
                )
            else:
                # Standard request with just the current prompt
                self.worker = OpenAIRequestWorker(
                    provider_config["api_url"], 
                    self.selected_model, 
                    final_prompt, 
                    self.current_api_keys.get("openai", ""),
                    stream=streaming_enabled
                )
        elif self.selected_provider == "anthropic":
            # Handle Anthropic conversation history
            if self.memory_enabled and self.conversation_history:
                # Include conversation history in messages
                messages = self.conversation_history.copy()
                messages.append(user_message)
                self.worker = AnthropicRequestWorker(
                    provider_config["api_url"], 
                    self.selected_model, 
                    messages, 
                    self.current_api_keys.get("anthropic", ""),
                    stream=streaming_enabled,
                    use_conversation=True  # Flag to indicate we're using conversation history
                )
            else:
                # Standard request with just the current prompt
                self.worker = AnthropicRequestWorker(
                    provider_config["api_url"], 
                    self.selected_model, 
                    final_prompt, 
                    self.current_api_keys.get("anthropic", ""),
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
                final_prompt,  # We'll include history directly in the prompt for Ollama
                stream=streaming_enabled
            )
        else:
            # Use generic handler for other providers
            self.worker = RequestWorker(
                provider_config, 
                self.selected_model, 
                final_prompt,  # We'll include history directly in the prompt for generic providers
                api_key=self.current_api_keys.get(self.selected_provider),
                stream=streaming_enabled
            )
        
        # Common worker setup
        self.worker.finished.connect(self.handle_response)
        self.worker.chunk_received.connect(self.handle_chunk)
        self.worker.start()

    def eventFilter(self, source, event):
        # Allow sending with Ctrl+Enter
        if (event.type() == event.KeyPress and 
            source is self.prompt_input and 
            event.key() == Qt.Key_Return and 
            event.modifiers() & Qt.ControlModifier):
            self.send_prompt()
            return True
        return super().eventFilter(source, event)
    
    def format_message_with_code_blocks(self, content, is_user_message=False):
        """Helper to format messages with proper code block handling"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # If this is a user message, format it differently
        if is_user_message:
            format = QTextCharFormat()
            format.setFontWeight(QFont.Normal)
            cursor.insertText(f"> {content}\n", format)
            return
        
        # For assistant messages, handle code blocks
        if "```" in content:
            # Split by code blocks
            pattern = r'(```(?:\w*)\n[\s\S]*?\n```)'
            parts = re.split(pattern, content)
            
            for part in parts:
                if part.strip() and part.startswith("```") and part.endswith("```"):
                    # This is a code block - format it properly
                    self.chat_display._insert_code_block(cursor, part)
                elif part.strip():
                    # Normal text part
                    format = QTextCharFormat()
                    format.setForeground(QColor("#24292e"))
                    cursor.insertText(part, format)
            
            # Add final newline
            cursor.insertBlock()
        else:
            # No code blocks, just insert normal text
            format = QTextCharFormat()
            format.setForeground(QColor("#24292e"))
            cursor.insertText(content + "\n", format)
        
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()
    
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
            # Store in cache
            self.cached_models["openai"] = models.copy()
        
            for model in models:
                self.model_dropdown.addItem(model)
            
            # Update AI-to-AI panel if it exists
            if hasattr(self, 'ai_to_ai_panel'):
                self.ai_to_ai_panel.update_provider_models("openai", models)
            
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
            # Store in cache
            self.cached_models["ollama"] = models.copy()

            for model in models:
                self.model_dropdown.addItem(model)
                
            # Update AI-to-AI panel if it exists
            if hasattr(self, 'ai_to_ai_panel'):
                self.ai_to_ai_panel.update_provider_models("ollama", models)
            
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
            # Store in cache
            self.cached_models["anthropic"] = models.copy()
            
            for model in models:
                self.model_dropdown.addItem(model)
            
            # Update AI-to-AI panel if it exists
            if hasattr(self, 'ai_to_ai_panel'):
                self.ai_to_ai_panel.update_provider_models("anthropic", models)
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
                if "<think>" in chunk_text:
                    self.in_think_section = True
                
                # Skip this chunk if we're in a thinking section
                if self.in_think_section:
                    if "</think>" in chunk_text:
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
            # Add a newline after the response
            self.chat_display.append("\n")
        elif success and not self.stream_checkbox.isChecked():
            # For non-streaming successful responses
            self.append_to_chat(response)
            # Add a newline after the response
            self.chat_display.append("\n")
        
        # NEW: Save assistant response to the database
        if success and self.current_thread_id:
            self.db_manager.add_message(
                self.current_thread_id,
                "assistant",
                response,
                provider=self.selected_provider,
                model=self.selected_model
            )
        
        # Add model attribution
        self.append_to_chat(f"[SYSTEM] Response from: {self.selected_provider.capitalize()} - {self.selected_model}")


        # NEW: Add assistant response to conversation history if memory is enabled
        if success and self.memory_enabled:
            # Get the latest user message
            latest_user_message = None
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                latest_user_message = self.conversation_history[-1]
            else:
                # Find the most recent user message
                for msg in reversed(self.conversation_history):
                    if msg["role"] == "user":
                        latest_user_message = msg
                        break
                        
            # Add both the user message (if not already added) and the assistant response
            if latest_user_message is None:
                # Get the user prompt from the chat display
                chat_text = self.chat_display.toPlainText()
                last_prompt = ""
                for line in reversed(chat_text.split("\n")):
                    if line.startswith(">"):
                        last_prompt = line[2:].strip()  # Remove the '> ' prefix
                        self.conversation_history.append({"role": "user", "content": last_prompt})
                        break
            
            # Add the assistant response
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Limit history to last 10 exchanges (20 messages) to prevent context overflow
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
        
        # Save this as the last used model for this provider
        self.last_used_models[self.selected_provider] = self.selected_model
        
        # Update the thread's provider and model if they've changed
        if self.current_thread_id:
            self.db_manager.update_thread(
                self.current_thread_id,
                provider=self.selected_provider,
                model=self.selected_model
            )
        
        # Reset processing state
        self.is_processing = False
        self.update_ui_state(enabled=True)
    
    def append_to_chat(self, text):
        """Add text to chat display and scroll to bottom"""
        
         # Ensure text ends with a newline
        if text and not text.endswith('\n'):
            text += '\n'
            
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
        self.memory_checkbox.setEnabled(enabled)  # NEW: Enable/disable memory checkbox
        
        # Server URL input (only for Ollama)
        if hasattr(self, 'server_url_input'):
            self.server_url_input.setEnabled(enabled)
    
    def clear_chat(self):
        """Clear the chat history"""
        # Create a custom message box with three buttons
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Clear Chat")
        msg_box.setText("Do you want to create a new thread or clear the current one?")
        
        # Add custom buttons
        new_thread_btn = msg_box.addButton("New Thread", QMessageBox.ActionRole)
        clear_current_btn = msg_box.addButton("Clear Current", QMessageBox.ActionRole)
        cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
        
        # Show the message box and get result
        msg_box.exec_()
        
        # Check which button was clicked
        clicked_button = msg_box.clickedButton()
        
        if clicked_button == cancel_btn:
            return
        
        if clicked_button == new_thread_btn:
            # Create a new thread with a default title
            title = "New Chat"
            self.current_thread_id = self.db_manager.create_thread(
                title,
                provider=self.selected_provider,
                model=self.selected_model
            )
            
            # Clear display
            self.chat_display.clear()
            self.chat_display.setText(f"[SYSTEM] New chat thread created. Provider: {PROVIDERS[self.selected_provider]['name']}, Model: {self.selected_model}\n\n")
            
            # Clear conversation history
            self.conversation_history = []
            
        elif clicked_button == clear_current_btn:
            if self.current_thread_id:
                reply = QMessageBox.question(self, "Confirm", 
                                          "Are you sure you want to delete all messages in this thread?",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    # Delete all messages but keep the thread
                    conn = sqlite3.connect(self.db_manager.db_path)
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM messages WHERE thread_id = ?', (self.current_thread_id,))
                    conn.commit()
                    conn.close()
                    
                    # Clear display
                    self.chat_display.clear()
                    provider_name = PROVIDERS[self.selected_provider]["name"]
                    self.chat_display.setText(f"[SYSTEM] Chat cleared. Provider: {provider_name}, Model: {self.selected_model}\n\n")
                    
                    # Clear conversation history
                    self.conversation_history = []
            else:
                # No current thread, just clear the display
                self.chat_display.clear()
                provider_name = PROVIDERS[self.selected_provider]["name"]
                self.chat_display.setText(f"[SYSTEM] Chat cleared. Provider: {provider_name}, Model: {self.selected_model}\n\n")
                
                # Clear conversation history
                self.conversation_history = []
        
        logging.debug("Chat cleared")
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # New thread action
        new_thread_action = QAction("New Thread", self)
        new_thread_action.setShortcut("Ctrl+N")
        new_thread_action.triggered.connect(self.create_new_thread)
        file_menu.addAction(new_thread_action)
        
        # Advanced new thread action
        # advanced_new_thread_action = QAction("Advanced New Thread...", self)
        # advanced_new_thread_action.triggered.connect(self.create_advanced_new_thread)
        # file_menu.addAction(advanced_new_thread_action)
        
        file_menu.addSeparator()
        
        # Import thread action
        import_thread_action = QAction("Import Thread...", self)
        import_thread_action.triggered.connect(self.import_thread)
        file_menu.addAction(import_thread_action)
        
        # Export thread action
        export_thread_action = QAction("Export Thread...", self)
        export_thread_action.triggered.connect(self.export_thread)
        file_menu.addAction(export_thread_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Thread menu
        thread_menu = menubar.addMenu("&Thread")
        
        # Rename thread action
        rename_thread_action = QAction("Rename Thread...", self)
        rename_thread_action.triggered.connect(self.rename_current_thread)
        thread_menu.addAction(rename_thread_action)
        
        # Archive thread action
        self.archive_thread_action = QAction("Archive Thread", self)
        self.archive_thread_action.triggered.connect(lambda: self.toggle_thread_archive(True))
        thread_menu.addAction(self.archive_thread_action)
        
        # Unarchive thread action
        self.unarchive_thread_action = QAction("Unarchive Thread", self)
        self.unarchive_thread_action.triggered.connect(lambda: self.toggle_thread_archive(False))
        self.unarchive_thread_action.setVisible(False)  # Hide initially
        thread_menu.addAction(self.unarchive_thread_action)
        
        thread_menu.addSeparator()
        
        # Delete thread action
        delete_thread_action = QAction("Delete Thread", self)
        delete_thread_action.triggered.connect(self.delete_current_thread)
        thread_menu.addAction(delete_thread_action)

    def create_new_thread(self):
        """Create a new chat thread"""
        title, ok = QInputDialog.getText(self, "New Chat", "Enter a title for this chat:")
        if ok and title:
            thread_id = self.db_manager.create_thread(title)
            if thread_id:
                self.load_thread(thread_id)

    def create_advanced_new_thread(self):
        """Create a new thread with advanced options"""
        dialog = ThreadCreationDialog(PROVIDERS, self)
        if dialog.exec_():
            values = dialog.get_values()
            
            thread_id = self.db_manager.create_thread(
                values["title"],
                provider=values["provider"],
                model=values["model"],
                preprompt=values["preprompt"]
            )
            
            if thread_id and values["preprompt"]:
                # Add the preprompt as a system message
                self.db_manager.add_message(
                    thread_id,
                    "system",
                    values["preprompt"]
                )
            
            if thread_id:
                self.load_thread(thread_id)

    def rename_current_thread(self):
        """Rename the current thread"""
        if not self.current_thread_id:
            QMessageBox.information(self, "No Thread", "No active thread to rename.")
            return
            
        thread = self.db_manager.get_thread(self.current_thread_id)
        if not thread:
            return
            
        title, ok = QInputDialog.getText(self, "Rename Thread", 
                                      "Enter new title:", 
                                      text=thread.get("title", ""))
        if ok and title:
            self.db_manager.update_thread(self.current_thread_id, title=title)
            
            # Update window title
            self.setWindowTitle(f"Multi-Provider AI Chat - {title}")

    def toggle_thread_archive(self, archive):
        """Archive or unarchive the current thread"""
        if not self.current_thread_id:
            QMessageBox.information(self, "No Thread", "No active thread to archive.")
            return
            
        self.db_manager.update_thread(self.current_thread_id, is_archived=archive)
        
        # Update UI
        self.archive_thread_action.setVisible(not archive)
        self.unarchive_thread_action.setVisible(archive)
        
        # Show indication in the window title
        thread = self.db_manager.get_thread(self.current_thread_id)
        if thread:
            title = thread.get("title", "")
            if archive:
                self.setWindowTitle(f"Multi-Provider AI Chat - {title} [Archived]")
            else:
                self.setWindowTitle(f"Multi-Provider AI Chat - {title}")

    def delete_current_thread(self):
        """Delete the current thread after confirmation"""
        if not self.current_thread_id:
            QMessageBox.information(self, "No Thread", "No active thread to delete.")
            return
            
        reply = QMessageBox.question(self, "Delete Thread", 
                                   "Are you sure you want to delete this thread? This cannot be undone.",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.db_manager.delete_thread(self.current_thread_id)
            
            # Clear display
            self.chat_display.clear()
            self.chat_display.setText("Thread deleted. Create or select another thread to continue.")
            
            # Reset current thread
            self.current_thread_id = None
            self.conversation_history = []
            
            # Reset window title
            self.setWindowTitle("Multi-Provider AI Chat")

    def import_thread(self):
        """Import a thread from a JSON file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Import Thread", 
                                               "", "JSON Files (*.json)")
        
        if filename:
            thread_id = self.db_manager.import_thread(filename)
            if thread_id:
                QMessageBox.information(self, "Import Successful", 
                                     "Thread imported successfully.")
                self.load_thread(thread_id)
            else:
                QMessageBox.warning(self, "Import Failed", 
                                 "Failed to import thread. Check the file format and see logs for details.")

    def export_thread(self):
        """Export the current thread to a JSON file"""
        if not self.current_thread_id:
            QMessageBox.information(self, "No Thread", "No active thread to export.")
            return
            
        thread = self.db_manager.get_thread(self.current_thread_id)
        if not thread:
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Export Thread", 
                                               f"{thread.get('title', 'chat')}.json", 
                                               "JSON Files (*.json)")
        
        if filename:
            success = self.db_manager.export_thread(self.current_thread_id, filename)
            if success:
                QMessageBox.information(self, "Export Successful", 
                                     f"Thread exported to {filename}")
            else:
                QMessageBox.warning(self, "Export Failed", 
                                 "Failed to export thread. See logs for details.")

    # Add method to handle the thread archive state UI update
    def update_thread_archive_actions(self, thread_id):
        """Update the archive/unarchive actions based on thread state"""
        if not thread_id:
            self.archive_thread_action.setVisible(False)
            self.unarchive_thread_action.setVisible(False)
            return
            
        thread = self.db_manager.get_thread(thread_id)
        if thread:
            is_archived = thread.get("is_archived", 0)
            self.archive_thread_action.setVisible(not is_archived)
            self.unarchive_thread_action.setVisible(is_archived)
        
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
        
        # If we have a current thread, save its ID in settings
        if self.current_thread_id:
            self.settings.setValue("last_thread_id", self.current_thread_id)
        
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