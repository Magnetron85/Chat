from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                            QPushButton, QInputDialog, QMessageBox, QMenu, QAction, QLineEdit,
                            QLabel, QSplitter, QFrame, QToolBar, QFileDialog, QDialog, QDialogButtonBox,
                            QFormLayout, QLineEdit, QTextEdit, QCheckBox, QComboBox, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QIcon, QFont, QColor, QTextDocument

from ai2ai_conversation_worker import AI2AIConversationWorker

class ThreadListWidget(QWidget):
    """Widget for displaying and managing the list of chat threads"""
    
    thread_selected = pyqtSignal(int)  # Emitted when a thread is selected
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.init_ui()
        
        # Connect signals from database manager
        self.db_manager.thread_list_changed.connect(self.update_thread_list)
        
        # Load initial list
        self.update_thread_list(self.db_manager.get_all_threads())
        
        # Search-as-you-type timer
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.perform_search)
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Thread list toolbar
        toolbar = QHBoxLayout()
        
        # New thread button
        self.new_btn = QPushButton("New Chat")
        self.new_btn.clicked.connect(self.create_new_thread)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search threads...")
        self.search_input.textChanged.connect(self.search_text_changed)
        
        toolbar.addWidget(self.new_btn)
        toolbar.addWidget(self.search_input)
        
        layout.addLayout(toolbar)
        
        # Thread list
        self.thread_list = QListWidget()
        self.thread_list.setSelectionMode(QListWidget.SingleSelection)
        self.thread_list.itemClicked.connect(self.on_thread_clicked)
        self.thread_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.thread_list.customContextMenuRequested.connect(self.show_context_menu)
        
        # Style the list
        self.thread_list.setStyleSheet("""
            QListWidget {
                background-color: #f7f7f7;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
           QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
                color: #000000;  /* Force black text for all items */
            }
            QListWidget::item:selected {
                background-color: #e0e0e0;
                color: #000000;  /* Keep black text when selected */
            }
        """)
        
        layout.addWidget(self.thread_list)
        
        self.setLayout(layout)
    
    def search_text_changed(self, text):
        """Handle search input changes with debouncing"""
        # Cancel any pending search
        self.search_timer.stop()
        
        if text:
            # Start a new timer (300ms debounce)
            self.search_timer.start(300)
        else:
            # If search cleared, immediately show all threads
            self.update_thread_list(self.db_manager.get_all_threads())
    
    def perform_search(self):
        """Execute the actual search after debounce timer expires"""
        search_text = self.search_input.text().strip()
        if search_text:
            # Search both thread titles and message content
            results = self.db_manager.search_threads(search_text)
            self.update_thread_list(results)
    
    def update_thread_list(self, threads):
        """Update the thread list with the provided threads"""
        self.thread_list.clear()
        
        if not threads:
            # Add a placeholder item if no threads
            item = QListWidgetItem("No chats yet. Click 'New Chat' to start.")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.thread_list.addItem(item)
            return
        
        # Inside the update_thread_list method:
        for thread in threads:
            # Format the thread item
            title = thread.get("title", "Untitled Chat")
            provider = thread.get("provider", "")
            model = thread.get("model", "")
            last_updated = thread.get("last_updated", "")
            message_count = thread.get("message_count", 0)
            
            # Format the timestamp for display
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(last_updated)
                formatted_time = dt.strftime("%m/%d/%Y %I:%M %p")
            except:
                formatted_time = last_updated
            
            # Create display text
            if thread.get("found_by_content"):
                matching_count = thread.get("matching_messages", 0)
                if provider and model:
                    display_text = f"{title} ({matching_count} matches)\n{provider} - {model} • {message_count} messages • {formatted_time}"
                else:
                    display_text = f"{title} ({matching_count} matches)\n{message_count} messages • {formatted_time}"
            else:
                if provider and model:
                    display_text = f"{title}\n{provider} - {model} • {message_count} messages • {formatted_time}"
                else:
                    display_text = f"{title}\n{message_count} messages • {formatted_time}"
            
            # Create the item
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, thread.get("id"))
            
            # Style archived threads differently
            if thread.get("is_archived", 0):
                item.setForeground(QColor("#888888"))
                item.setToolTip("Archived chat")
            
            self.thread_list.addItem(item)
    
    def on_thread_clicked(self, item):
        """Handle thread selection"""
        thread_id = item.data(Qt.UserRole)
        if thread_id:
            self.thread_selected.emit(thread_id)
    
    def create_new_thread(self):
        """Create a new chat thread"""
        title, ok = QInputDialog.getText(self, "New Chat", "Enter a title for this chat:")
        if ok and title:
            thread_id = self.db_manager.create_thread(title)
            if thread_id:
                self.thread_selected.emit(thread_id)
                
                # Select the newly created thread in the list
                for i in range(self.thread_list.count()):
                    item = self.thread_list.item(i)
                    if item.data(Qt.UserRole) == thread_id:
                        self.thread_list.setCurrentItem(item)
                        break
    
    def show_context_menu(self, position):
        """Show context menu for thread items"""
        item = self.thread_list.itemAt(position)
        if not item:
            return
            
        thread_id = item.data(Qt.UserRole)
        if not thread_id:
            return
            
        # Get thread info
        thread = self.db_manager.get_thread(thread_id)
        if not thread:
            return
            
        # Create context menu
        menu = QMenu()
        
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self.rename_thread(thread_id))
        menu.addAction(rename_action)
        
        # Toggle archive action
        if thread.get("is_archived", 0):
            archive_action = QAction("Unarchive", self)
            archive_action.triggered.connect(lambda: self.toggle_thread_archive(thread_id, False))
        else:
            archive_action = QAction("Archive", self)
            archive_action.triggered.connect(lambda: self.toggle_thread_archive(thread_id, True))
        menu.addAction(archive_action)
        
        # Export action
        export_action = QAction("Export", self)
        export_action.triggered.connect(lambda: self.export_thread(thread_id))
        menu.addAction(export_action)
        
        menu.addSeparator()
        
        # Delete action
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self.delete_thread(thread_id))
        menu.addAction(delete_action)
        
        # Show the menu
        menu.exec_(self.thread_list.mapToGlobal(position))
    
    def rename_thread(self, thread_id):
        """Rename a thread"""
        thread = self.db_manager.get_thread(thread_id)
        if not thread:
            return
            
        title, ok = QInputDialog.getText(self, "Rename Chat", 
                                        "Enter new title:", 
                                        text=thread.get("title", ""))
        if ok and title:
            self.db_manager.update_thread(thread_id, title=title)
    
    def toggle_thread_archive(self, thread_id, archive):
        """Archive or unarchive a thread"""
        self.db_manager.update_thread(thread_id, is_archived=archive)
    
    def delete_thread(self, thread_id):
        """Delete a thread after confirmation"""
        reply = QMessageBox.question(self, "Delete Chat", 
                                   "Are you sure you want to delete this chat? This cannot be undone.",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.db_manager.delete_thread(thread_id)
    
    def export_thread(self, thread_id):
        """Export a thread to a JSON file"""
        thread = self.db_manager.get_thread(thread_id)
        if not thread:
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Export Chat", 
                                                f"{thread.get('title', 'chat')}.json", 
                                                "JSON Files (*.json)")
        
        if filename:
            success = self.db_manager.export_thread(thread_id, filename)
            if success:
                QMessageBox.information(self, "Export Successful", 
                                      f"Chat exported to {filename}")
            else:
                QMessageBox.warning(self, "Export Failed", 
                                  "Failed to export chat. See logs for details.")


class ThreadCreationDialog(QDialog):
    """Dialog for creating a new chat thread with advanced options"""
    
    def __init__(self, providers, parent=None):
        super().__init__(parent)
        self.providers = providers
        self.setWindowTitle("Create New Chat")
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Form layout for inputs
        form = QFormLayout()
        
        # Title input
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("My new chat")
        form.addRow("Chat Title:", self.title_input)
        
        # Provider selection
        self.provider_combo = QComboBox()
        for provider_id, config in self.providers.items():
            self.provider_combo.addItem(config["name"], provider_id)
        form.addRow("Default Provider:", self.provider_combo)
        
        # Model selection (will be populated when provider changes)
        self.model_combo = QComboBox()
        form.addRow("Default Model:", self.model_combo)
        
        # Connect provider change to update models
        self.provider_combo.currentIndexChanged.connect(self.update_model_combo)
        
        # Initial model population
        self.update_model_combo()
        
        # Preprompt input
        self.preprompt_input = QTextEdit()
        self.preprompt_input.setPlaceholderText("Optional: Enter system instructions for the AI...")
        self.preprompt_input.setMaximumHeight(100)
        form.addRow("Preprompt:", self.preprompt_input)
        
        layout.addLayout(form)
        
        # Dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
        self.setLayout(layout)
        self.resize(400, 300)
    
    def update_model_combo(self):
        """Update the model dropdown based on selected provider"""
        self.model_combo.clear()
        
        provider_id = self.provider_combo.currentData()
        if not provider_id or provider_id not in self.providers:
            return
            
        provider_config = self.providers[provider_id]
        
        if "models" in provider_config:
            for model in provider_config["models"]:
                self.model_combo.addItem(model, model)
    
    def get_values(self):
        """Get the dialog values"""
        return {
            "title": self.title_input.text(),
            "provider": self.provider_combo.currentData(),
            "model": self.model_combo.currentText(),
            "preprompt": self.preprompt_input.toPlainText()
        }


class SearchResultsWidget(QWidget):
    """Widget for displaying search results across threads"""
    
    result_selected = pyqtSignal(int, int)  # thread_id, message_id
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.init_ui()
        
        # Connect to database search results signal
        self.db_manager.search_results_changed.connect(self.display_results)
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header
        self.header_label = QLabel("Search Results")
        self.header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.header_label)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search all chats...")
        self.search_input.textChanged.connect(self.on_search_text_changed)
        layout.addWidget(self.search_input)
        
        # Results list
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_result_clicked)
        self.results_list.setTextElideMode(Qt.ElideNone)
        self.results_list.setWordWrap(True)
        self.results_list.setStyleSheet("""
            QListWidget {
                background-color: #f7f7f7;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e0e0e0;
                color: #333;
            }
        """)
        layout.addWidget(self.results_list)
        
        self.setLayout(layout)
        
        # Search debounce timer
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.perform_search)
        
    def on_search_text_changed(self, text):
        """Handle search text changes with debouncing"""
        # Reset timer on each keystroke
        self.search_timer.stop()
        
        if not text:
            # Clear results if search is empty
            self.results_list.clear()
            return
        
        # Set timer for 300ms debounce
        self.search_timer.start(300)
    
    def perform_search(self):
        """Execute search after debounce delay"""
        search_text = self.search_input.text().strip()
        if search_text:
            logging.debug(f"SearchResultsWidget performing search for: '{search_text}'")
            results = self.db_manager.search_messages(search_text)
            # Ensure we have results by directly accessing them, not relying solely on the signal
            if results:
                self.display_results(results)
            else:
                logging.debug("No results returned from search_messages")
                # Display empty results as a fallback
                self.results_list.clear()
                item = QListWidgetItem("No results found")
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                self.results_list.addItem(item)
    
    def display_results(self, results):
        """Display search results"""
        self.results_list.clear()
        
        if not results:
            item = QListWidgetItem("No results found")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.results_list.addItem(item)
            return
        
        # Update header
        self.header_label.setText(f"Search Results ({len(results)})")
        
        # Group results by thread
        thread_results = {}
        for result in results:
            thread_id = result.get("thread_id")
            if thread_id not in thread_results:
                thread_results[thread_id] = {
                    "title": result.get("thread_title", "Unknown"),
                    "messages": []
                }
            thread_results[thread_id]["messages"].append(result)
        
        # Add results by thread
        for thread_id, thread_data in thread_results.items():
            # Add thread header
            thread_header = QListWidgetItem(f"▼ {thread_data['title']} ({len(thread_data['messages'])} matches)")
            thread_header.setData(Qt.UserRole, {"type": "thread_header", "thread_id": thread_id})
            thread_header.setBackground(QColor("#e0e0e0"))
            font = thread_header.font()
            font.setBold(False)
            thread_header.setFont(font)
            self.results_list.addItem(thread_header)
            
            # Add message results
            for message in thread_data["messages"]:
                # Format snippet (max 100 chars)
                content = message.get("content", "")
                search_term = self.search_input.text().strip()
                
                # Find position of search term
                pos = content.lower().find(search_term.lower())
                if pos >= 0:
                    # Create snippet centered around search term
                    start = max(0, pos - 40)
                    end = min(len(content), pos + len(search_term) + 40)
                    
                    snippet = content[start:end]
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet = snippet + "..."
                    
                    # Find where the term appears in our snippet
                    snippet_pos = snippet.lower().find(search_term.lower())
                    if snippet_pos >= 0:
                        # Use visible special characters to highlight the match
                        highlighted_snippet = (
                            snippet[:snippet_pos] + 
                            "【" + snippet[snippet_pos:snippet_pos+len(search_term)] + "】" + 
                            snippet[snippet_pos+len(search_term):]
                        )
                    else:
                        highlighted_snippet = snippet
                else:
                    # Fallback to first 100 chars
                    snippet = content[:100] + ("..." if len(content) > 100 else "")
                    highlighted_snippet = snippet
                
                # Format role
                role = message.get("role", "user")
                if role == "user":
                    role_display = "You"
                elif role == "assistant":
                    role_display = "AI"
                else:
                    role_display = role.capitalize()
                
                # Format timestamp
                timestamp = message.get("timestamp", "")
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp)
                    timestamp_display = dt.strftime("%m/%d/%Y %I:%M %p")
                except:
                    timestamp_display = timestamp
                
                # Create display text with visible highlighting
                display_text = f"{role_display}: {highlighted_snippet}\n{timestamp_display}"
                
                # Create list item (no HTML needed)
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, {
                    "type": "message",
                    "thread_id": thread_id,
                    "message_id": message.get("id")
                })
                item.setIndent(10)  # Indent to show hierarchy
                
                self.results_list.addItem(item)
            
            # Add spacer item
            spacer = QListWidgetItem("")
            spacer.setFlags(Qt.NoItemFlags)
            self.results_list.addItem(spacer)
            
    def highlight_text(self, text, search_term):
        """Highlight search term occurrences in text with HTML"""
        if not search_term:
            return text
        
        pos = text.lower().find(search_term.lower())
        if pos >= 0:
            return text[:pos] + f"<span style='background-color: #FFFF77;'>{text[pos:pos+len(search_term)]}</span>" + self.highlight_text(text[pos+len(search_term):], search_term)
        return text
    
    def on_result_clicked(self, item):
        """Handle result item click"""
        data = item.data(Qt.UserRole)
        if not data:
            return
            
        if data.get("type") == "message":
            thread_id = data.get("thread_id")
            message_id = data.get("message_id")
            if thread_id and message_id:
                self.result_selected.emit(thread_id, message_id)
        elif data.get("type") == "thread_header":
            thread_id = data.get("thread_id")
            if thread_id:
                self.result_selected.emit(thread_id, None)


class AIToChatPanel(QWidget):
    """Panel for initiating AI-to-AI conversations"""
    
    def __init__(self, db_manager, providers, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.providers = providers
        self.init_ui()
        self.ai2ai_worker = None
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("AI-to-AI Conversation")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)
        
        # Description
        description = QLabel(
            "Set up a conversation between two AI models. "
            "They will exchange messages until a solution is reached "
            "or the maximum number of turns is reached."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Settings form
        form = QFormLayout()
        
        # Title
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("AI Discussion Session")
        form.addRow("Thread Title:", self.title_input)
        
        # AI 1 settings
        self.ai1_provider = QComboBox()
        self.ai1_model = QComboBox()
        for provider_id, config in self.providers.items():
            self.ai1_provider.addItem(config["name"], provider_id)
        
        self.ai1_provider.currentIndexChanged.connect(lambda: self.update_model_combo(self.ai1_provider, self.ai1_model))
        form.addRow("AI 1 Provider:", self.ai1_provider)
        form.addRow("AI 1 Model:", self.ai1_model)
        
        # AI 1 system prompt
        self.ai1_prompt = QTextEdit()
        self.ai1_prompt.setPlaceholderText("Instructions for AI 1...")
        self.ai1_prompt.setMaximumHeight(80)
        form.addRow("AI 1 Instructions:", self.ai1_prompt)
        
        # AI 2 settings
        self.ai2_provider = QComboBox()
        self.ai2_model = QComboBox()
        for provider_id, config in self.providers.items():
            self.ai2_provider.addItem(config["name"], provider_id)
        
        self.ai2_provider.currentIndexChanged.connect(lambda: self.update_model_combo(self.ai2_provider, self.ai2_model))
        form.addRow("AI 2 Provider:", self.ai2_provider)
        form.addRow("AI 2 Model:", self.ai2_model)
        
        # AI 2 system prompt
        self.ai2_prompt = QTextEdit()
        self.ai2_prompt.setPlaceholderText("Instructions for AI 2...")
        self.ai2_prompt.setMaximumHeight(80)
        form.addRow("AI 2 Instructions:", self.ai2_prompt)
        
        # Task prompt
        self.task_prompt = QTextEdit()
        self.task_prompt.setPlaceholderText("Describe the problem or task for the AIs to discuss...")
        form.addRow("Task Description:", self.task_prompt)
        
        # Initial context (optional)
        self.initial_context = QTextEdit()
        self.initial_context.setPlaceholderText("Optional: Provide background information or context for the conversation...")
        self.initial_context.setMaximumHeight(80)
        form.addRow("Initial Context:", self.initial_context)
        
        # Conversation style
        self.conversation_style = QComboBox()
        self.conversation_style.addItem("Standard", "standard")
        self.conversation_style.addItem("Debate", "debate")
        self.conversation_style.addItem("Collaborative", "collaborative")
        self.conversation_style.addItem("Detailed", "detailed")
        form.addRow("Conversation Style:", self.conversation_style)
        
        # Max turns
        self.max_turns = QComboBox()
        for i in range(1, 11):
            self.max_turns.addItem(str(i), i)
        self.max_turns.setCurrentIndex(4)  # Default to 5 turns
        form.addRow("Maximum Turns:", self.max_turns)
        
        # Include thinking sections checkbox
        self.include_thinking = QCheckBox("Include thinking sections in responses (if supported)")
        form.addRow("", self.include_thinking)
        
        layout.addLayout(form)
        
        # Progress bar and status
        self.progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_layout.addWidget(self.progress_label)
        self.progress_layout.addWidget(self.progress_bar)
        layout.addLayout(self.progress_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Conversation")
        self.start_btn.clicked.connect(self.start_conversation)
        
        self.stop_btn = QPushButton("Stop Conversation")
        self.stop_btn.clicked.connect(self.stop_conversation)
        self.stop_btn.setEnabled(False)
        
        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Initial model combo population
        self.update_model_combo(self.ai1_provider, self.ai1_model)
        self.update_model_combo(self.ai2_provider, self.ai2_model)
    
    def update_model_combo(self, provider_combo, model_combo):
        """Update a model combo box based on selected provider"""
        model_combo.clear()
        
        provider_id = provider_combo.currentData()
        if not provider_id or provider_id not in self.providers:
            return
            
        provider_config = self.providers[provider_id]
        
        # Check if we can get cached models from the parent window
        parent_window = self.window()
        if hasattr(parent_window, 'cached_models') and provider_id in parent_window.cached_models:
            cached_models = parent_window.cached_models[provider_id]
            if cached_models:
                # Use cached models if available
                for model in cached_models:
                    model_combo.addItem(model, model)
                return
        
        # Fall back to predefined models if available
        if "models" in provider_config:
            for model in provider_config["models"]:
                model_combo.addItem(model, model)
        else:
            # For dynamic providers (like Ollama) without cached models
            if provider_id == "ollama":
                model_combo.addItem("Please load models in Chat tab first", "")
            elif provider_id in ["openai", "anthropic"]:
                model_combo.addItem("Please configure API key and load models in Chat tab first", "")
            
    def update_provider_models(self, provider_id, models):
        """Update models for a provider from the main application cache"""
        if provider_id == "ollama":
            if self.ai1_provider.currentData() == "ollama":
                self.update_combo_with_models(self.ai1_model, models)
            if self.ai2_provider.currentData() == "ollama":
                self.update_combo_with_models(self.ai2_model, models)
        
        elif provider_id == "openai":
            if self.ai1_provider.currentData() == "openai":
                self.update_combo_with_models(self.ai1_model, models)
            if self.ai2_provider.currentData() == "openai":
                self.update_combo_with_models(self.ai2_model, models)
        
        elif provider_id == "anthropic":
            if self.ai1_provider.currentData() == "anthropic":
                self.update_combo_with_models(self.ai1_model, models)
            if self.ai2_provider.currentData() == "anthropic":
                self.update_combo_with_models(self.ai2_model, models)

    def update_combo_with_models(self, combo_box, models):
        """Helper to update a combo box with models"""
        current_text = combo_box.currentText()
        combo_box.clear()
        
        for model in models:
            combo_box.addItem(model, model)
        
        # Try to restore previous selection if possible
        if current_text and current_text != "Loading models...":
            index = combo_box.findText(current_text)
            if index >= 0:
                combo_box.setCurrentIndex(index)
            
    def start_conversation(self):
        """Start an AI-to-AI conversation"""
        # Validate inputs
        title = self.title_input.text().strip()
        if not title:
            title = "AI Discussion Session"
            
        task = self.task_prompt.toPlainText().strip()
        if not task:
            QMessageBox.warning(self, "Missing Task", 
                               "Please describe the task or problem for the AIs to discuss.")
            return
            
        # Create a new thread
        thread_id = self.db_manager.create_thread(title)
        if not thread_id:
            QMessageBox.critical(self, "Error", "Failed to create thread.")
            return
            
        # Prepare AI configurations - without API keys
        ai1_config = {
            'provider': self.ai1_provider.currentData(),
            'model': self.ai1_model.currentText(),
            'prompt': self.ai1_prompt.toPlainText().strip()
        }
        
        ai2_config = {
            'provider': self.ai2_provider.currentData(),
            'model': self.ai2_model.currentText(),
            'prompt': self.ai2_prompt.toPlainText().strip()
        }
        
        # Get additional options
        max_turns = int(self.max_turns.currentText())
        include_thinking = self.include_thinking.isChecked()
        initial_context = self.initial_context.toPlainText().strip() or None
        conversation_style = self.conversation_style.currentData()
        
        # Create the AI2AI worker
        self.ai2ai_worker = AI2AIConversationWorker(
            self.db_manager,
            thread_id,
            ai1_config,
            ai2_config,
            task,
            max_turns,
            include_thinking=include_thinking,
            initial_context=initial_context,
            conversation_style=conversation_style
        )
        
        # Connect signals
        self.ai2ai_worker.message_added.connect(self.on_message_added)
        self.ai2ai_worker.conversation_finished.connect(self.on_conversation_finished)
        self.ai2ai_worker.progress_updated.connect(self.on_progress_updated)
        
        # Update UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max_turns)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting conversation...")
        
        # Toggle buttons
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Start the worker thread
        self.ai2ai_worker.start()
        
        # Show notification
        QMessageBox.information(self, "Conversation Started", 
                              f"AI-to-AI conversation has been initiated in thread '{title}'.")
        
        # Return the thread ID so the main app can switch to it
        return thread_id
    
    def stop_conversation(self):
        """Stop the ongoing AI-to-AI conversation"""
        if self.ai2ai_worker and self.ai2ai_worker.isRunning():
            self.progress_label.setText("Stopping conversation...")
            self.ai2ai_worker.stop()
    
    def on_message_added(self, role, content, model):
        """Handle new message added to conversation"""
        # This method is called when the worker adds a new message
        if role == "ai1":
            provider = self.ai1_provider.currentText()
            self.progress_label.setText(f"AI 1 ({model})")
            
            # Before adding the content, add a header message to indicate which AI is speaking
            self.db_manager.add_message(
                self.ai2ai_worker.thread_id,
                "system",
                f"AI 1 ({provider} - {model})"
            )
            
            # Add attribution message after AI1's message
            self.db_manager.add_message(
                self.ai2ai_worker.thread_id,
                "system",
                f"Response from: {provider} - {model}"
            )
        elif role == "ai2":
            provider = self.ai2_provider.currentText()
            self.progress_label.setText(f"AI 2 ({model})")
            
            # Before adding the content, add a header message to indicate which AI is speaking
            self.db_manager.add_message(
                self.ai2ai_worker.thread_id,
                "system",
                f"AI 2 ({provider} - {model})"
            )
            
            # Add attribution message after AI2's message
            self.db_manager.add_message(
                self.ai2ai_worker.thread_id,
                "system",
                f"Response from: {provider} - {model}"
            )
    
    def on_progress_updated(self, current_turn, max_turns):
        """Update progress bar based on current turn"""
        self.progress_bar.setRange(0, max_turns)
        self.progress_bar.setValue(current_turn)
        self.progress_label.setText(f"Turn {current_turn} of {max_turns}")
    
    def on_conversation_finished(self, success, message):
        """Handle conversation completion"""
        # Update UI
        self.progress_bar.setVisible(False)
        self.progress_label.setText(message)
        
        # Toggle buttons
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Clean up worker
        self.ai2ai_worker = None
        
        # Show notification
        QMessageBox.information(self, "Conversation Finished", message)
