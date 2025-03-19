import json
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QTextEdit, QComboBox, QInputDialog, QMessageBox,
                            QMenu, QAction, QWidget, QCheckBox)
from PyQt5.QtCore import Qt, QSettings

class PrepromptManager:
    """Handles storage, retrieval, and validation of preprompts"""
    
    def __init__(self, parent, settings):
        self.parent = parent
        self.settings = settings
        self.preprompts = {}
        self.usage_count = {}  # Track usage frequency
        self.current_preprompt = None
        self.default_preprompt = None
        self.use_last_as_default = True  # Default to using last selected preprompt
        self.load_preprompts()
        
    def load_preprompts(self):
        """Load saved preprompts from settings"""
        # Load default preprompt settings first
        self.default_preprompt = self.settings.value("default_preprompt", "")
        self.use_last_as_default = self.settings.value("use_last_as_default", True, type=bool)
    
        # Then load the preprompts...
        size = self.settings.beginReadArray("preprompts")
        self.preprompts = {}
        self.usage_count = {}  # Track usage frequency
        
        for i in range(size):
            self.settings.setArrayIndex(i)
            name = self.settings.value("name")
            text = self.settings.value("text")
            count = int(self.settings.value("usage_count", 0))  # Get usage count
            self.preprompts[name] = text
            self.usage_count[name] = count
        
        self.settings.endArray()
        
        # If no preprompts were loaded from array, try the old JSON format
        if size == 0:
            preprompts_json = self.settings.value("preprompts", "{}")
            try:
                self.preprompts = json.loads(preprompts_json)
                # Ensure it's a dict even if empty
                if not isinstance(self.preprompts, dict):
                    self.preprompts = {}
                # Initialize usage counts to 0
                for name in self.preprompts:
                    self.usage_count[name] = 0
            except:
                self.preprompts = {}
        
        # Load default preprompt settings
        self.default_preprompt = self.settings.value("default_preprompt", "")
        self.use_last_as_default = self.settings.value("use_last_as_default", True, type=bool)
        
        # Load current preprompt if any
        current_id = self.settings.value("current_preprompt", "")
        
        # Set the current preprompt based on settings
        if current_id and current_id in self.preprompts:
            self.current_preprompt = current_id
        elif self.use_last_as_default and current_id:
            # If use_last_as_default is enabled, try to use the last selected preprompt
            self.current_preprompt = current_id
        elif self.default_preprompt and self.default_preprompt in self.preprompts:
            # If a default is set and it exists, use it
            self.current_preprompt = self.default_preprompt
    
    def save_preprompts(self):
        """Save preprompts to settings"""
        self.settings.beginWriteArray("preprompts")
        
        i = 0
        for name, text in self.preprompts.items():
            self.settings.setArrayIndex(i)
            self.settings.setValue("name", name)
            self.settings.setValue("text", text)
            self.settings.setValue("usage_count", self.usage_count.get(name, 0))  # Save usage count
            i += 1
        
        self.settings.endArray()
        
        self.settings.setValue("default_preprompt", self.default_preprompt)
        self.settings.setValue("use_last_as_default", self.use_last_as_default)
        
        if self.current_preprompt:
            self.settings.setValue("current_preprompt", self.current_preprompt)
        
        self.settings.setValue("default_preprompt", self.default_preprompt)
        self.settings.setValue("use_last_as_default", self.use_last_as_default)
        self.settings.sync()
    
    def increment_usage_count(self, name):
        """Increment the usage count for a preprompt"""
        if name in self.preprompts:
            self.usage_count[name] = self.usage_count.get(name, 0) + 1
            self.save_preprompts()
    
    def add_preprompt(self, name, content):
        """Add or update a preprompt"""
        if not name:
            return False
            
        # Validate preprompt before saving
        if not self.validate_preprompt(content):
            return False
            
        self.preprompts[name] = content
        if name not in self.usage_count:
            self.usage_count[name] = 0
        self.save_preprompts()
        return True
    
    def remove_preprompt(self, name):
        """Remove a preprompt by name"""
        if name in self.preprompts:
            del self.preprompts[name]
            if name in self.usage_count:
                del self.usage_count[name]
            
            # Update current and default preprompt if they match the removed one
            if self.current_preprompt == name:
                self.current_preprompt = None
            
            if self.default_preprompt == name:
                self.default_preprompt = None
                
            self.save_preprompts()
            return True
        return False
    
    def set_default_preprompt(self, name):
        """Set the default preprompt"""
        if name in self.preprompts or name is None:
            self.default_preprompt = name
            self.save_preprompts()
            return True
        return False
    
    def set_use_last_as_default(self, value):
        """Set whether to use the last selected preprompt as default"""
        self.use_last_as_default = bool(value)
        self.save_preprompts()
    
    def validate_preprompt(self, text):
        """
        Validate a preprompt to ensure it doesn't break anything.
        This is a simple validation to check for obvious issues.
        """
        # Ensure it's a string
        if not isinstance(text, str):
            return False
            
        # Check if any unbalanced brackets or tags
        brackets = {
            '{': '}',
            '[': ']',
            '(': ')',
            '<': '>'
        }
        
        stack = []
        
        for char in text:
            if char in brackets.keys():
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    # Closing bracket without opening bracket
                    return False
                
                # Check if closing bracket matches the last opening bracket
                opening_bracket = stack.pop()
                if char != brackets[opening_bracket]:
                    return False
        
        # If stack is not empty, there are unclosed brackets
        if stack:
            return False
        
        return True
    
    def get_preprompt_by_name(self, name):
        """Get preprompt content by name"""
        return self.preprompts.get(name, "")
    
    def get_all_preprompt_names(self):
        """Get a list of all preprompt names"""
        return list(self.preprompts.keys())
    
    def get_all_preprompt_names_sorted_by_usage(self):
        """Get a list of all preprompt names sorted by usage count"""
        return sorted(
            self.preprompts.keys(),
            key=lambda x: self.usage_count.get(x, 0),
            reverse=True
        )
    
    def get_current_preprompt(self):
        """Get the currently selected preprompt content"""
        if self.current_preprompt:
            return self.preprompts.get(self.current_preprompt, "")
        return ""
    
    def set_current_preprompt(self, name):
        """Set the current preprompt by name"""
        if name == "None":
            self.current_preprompt = None
            self.save_preprompts()
            return True
        elif name in self.preprompts or name is None:
            self.current_preprompt = name
            self.save_preprompts()
            return True
        return False


class PrepromptUI:
    """UI components for the preprompt functionality"""
    
    def __init__(self, parent, preprompt_manager):
        self.parent = parent
        self.preprompt_manager = preprompt_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Create the UI components for preprompt management"""
        # Create preprompt group box
        self.preprompt_group = QGroupBox("Preprompt Manager")
        preprompt_layout = QVBoxLayout()
        
        # Selector and buttons layout
        selector_layout = QHBoxLayout()
        
        # Preprompt selector
        self.preprompt_label = QLabel("Preprompt:")
        self.preprompt_dropdown = QComboBox()
        self.preprompt_dropdown.setMinimumWidth(200)
        self.update_preprompt_dropdown()
        self.preprompt_dropdown.currentTextChanged.connect(self.on_preprompt_selected)
        
        # Buttons
        self.new_preprompt_btn = QPushButton("New")
        self.new_preprompt_btn.clicked.connect(self.create_new_preprompt)
        
        self.save_preprompt_btn = QPushButton("Save")
        self.save_preprompt_btn.clicked.connect(self.save_current_preprompt)
        
        self.delete_preprompt_btn = QPushButton("Delete")
        self.delete_preprompt_btn.clicked.connect(self.delete_current_preprompt)
        
        # Add default options button with menu
        self.default_options_btn = QPushButton("Default Options")
        self.default_options_btn.clicked.connect(self.show_default_options)
        
        selector_layout.addWidget(self.preprompt_label)
        selector_layout.addWidget(self.preprompt_dropdown, 1)
        selector_layout.addWidget(self.new_preprompt_btn)
        selector_layout.addWidget(self.save_preprompt_btn)
        selector_layout.addWidget(self.delete_preprompt_btn)
        selector_layout.addWidget(self.default_options_btn)
        
        # Preprompt text editor
        self.preprompt_editor = QTextEdit()
        self.preprompt_editor.setPlaceholderText("Enter preprompt text here...")
        
        # Load current preprompt if any
        current_preprompt = self.preprompt_manager.get_current_preprompt()
        if current_preprompt:
            self.preprompt_editor.setText(current_preprompt)
        
        # Default settings checkboxes
        default_settings_layout = QHBoxLayout()
        
        self.use_last_checkbox = QCheckBox("Use last selected preprompt as default")
        self.use_last_checkbox.setChecked(self.preprompt_manager.use_last_as_default)
        self.use_last_checkbox.stateChanged.connect(self.on_use_last_changed)
        
        default_settings_layout.addWidget(self.use_last_checkbox)
        default_settings_layout.addStretch()
        
        # Current default display
        self.default_label = QLabel()
        self.update_default_label()
        default_settings_layout.addWidget(self.default_label)
        
        # Add components to layout
        preprompt_layout.addLayout(selector_layout)
        preprompt_layout.addWidget(self.preprompt_editor)
        preprompt_layout.addLayout(default_settings_layout)
        
        # Add info text
        info_label = QLabel(
            "Preprompts are added at the beginning of your prompt to provide context to the AI model."
        )
        info_label.setWordWrap(True)
        preprompt_layout.addWidget(info_label)
        
        self.preprompt_group.setLayout(preprompt_layout)
    
    def update_preprompt_dropdown(self):
        """Update the preprompt dropdown with sorted preprompts"""
        current_text = self.preprompt_dropdown.currentText()
        
        self.preprompt_dropdown.clear()
        self.preprompt_dropdown.addItem("None")
        
        # Sort preprompts by usage count (descending)
        sorted_preprompts = self.preprompt_manager.get_all_preprompt_names_sorted_by_usage()
        for name in sorted_preprompts:
            self.preprompt_dropdown.addItem(name)
        
        # Restore selection if possible
        if self.preprompt_manager.current_preprompt:
            index = self.preprompt_dropdown.findText(self.preprompt_manager.current_preprompt)
            if index >= 0:
                self.preprompt_dropdown.setCurrentIndex(index)
        elif current_text:
            index = self.preprompt_dropdown.findText(current_text)
            if index >= 0:
                self.preprompt_dropdown.setCurrentIndex(index)
    
    def update_default_label(self):
        """Update the label showing the current default preprompt"""
        if self.preprompt_manager.use_last_as_default:
            self.default_label.setText("Default: Last selected preprompt")
        elif self.preprompt_manager.default_preprompt:
            self.default_label.setText(f"Default: {self.preprompt_manager.default_preprompt}")
        else:
            self.default_label.setText("No default preprompt set")
    
    def on_preprompt_selected(self, name):
        """Handle preprompt selection change"""
        if name == "None":
            self.preprompt_manager.set_current_preprompt("None")  # Pass "None" as string
            self.preprompt_editor.clear()
            return
            
        self.preprompt_manager.set_current_preprompt(name)
        content = self.preprompt_manager.get_preprompt_by_name(name)
        self.preprompt_editor.setText(content)
    
    def on_use_last_changed(self, state):
        """Handle change to 'use last selected' checkbox"""
        use_last = (state == Qt.Checked)
        self.preprompt_manager.set_use_last_as_default(use_last)
        self.update_default_label()
    
    def show_default_options(self):
        """Show a menu with default preprompt options"""
        menu = QMenu(self.parent)
        
        # Add option to set current as default
        current_text = self.preprompt_dropdown.currentText()
        if current_text != "None":
            set_current_action = QAction(f"Set '{current_text}' as default", self.parent)
            set_current_action.triggered.connect(lambda: self.set_default_preprompt(current_text))
            menu.addAction(set_current_action)
        
        # Add options for all preprompts
        if self.preprompt_manager.get_all_preprompt_names():
            menu.addSeparator()
            for name in self.preprompt_manager.get_all_preprompt_names():
                action = QAction(f"Set '{name}' as default", self.parent)
                action.triggered.connect(lambda checked, n=name: self.set_default_preprompt(n))
                menu.addAction(action)
        
        # Add option to clear default
        if self.preprompt_manager.default_preprompt:
            menu.addSeparator()
            clear_action = QAction("Clear default preprompt", self.parent)
            clear_action.triggered.connect(lambda: self.set_default_preprompt(None))
            menu.addAction(clear_action)
        
        # Show the menu
        menu.exec_(self.default_options_btn.mapToGlobal(self.default_options_btn.rect().bottomLeft()))
    
    def set_default_preprompt(self, name):
        """Set the specified preprompt as default"""
        if name == "None":
            # Pass None to the manager when "None" is selected in dropdown
            success = self.preprompt_manager.set_default_preprompt(None)
        else:
            success = self.preprompt_manager.set_default_preprompt(name)
        
        if success:
            # If setting a specific default, disable "use last" option
            if name is not None and name != "None":
                self.preprompt_manager.set_use_last_as_default(False)
                self.use_last_checkbox.setChecked(False)
            
            self.update_default_label()
            QMessageBox.information(
                self.parent, "Default Preprompt", 
                f"Default preprompt {'cleared' if name == 'None' or name is None else f'set to \'{name}\''}"
            )
    
    def create_new_preprompt(self):
        """Create a new preprompt"""
        name, ok = QInputDialog.getText(
            self.parent, "New Preprompt", "Enter name for the new preprompt:"
        )
        
        if ok and name:
            if name in self.preprompt_manager.get_all_preprompt_names():
                QMessageBox.warning(
                    self.parent, "Duplicate Name", 
                    f"A preprompt named '{name}' already exists."
                )
                return
                
            # Create empty preprompt and select it
            self.preprompt_manager.add_preprompt(name, "")
            self.update_preprompt_dropdown()
            index = self.preprompt_dropdown.findText(name)
            if index >= 0:
                self.preprompt_dropdown.setCurrentIndex(index)
    
    def save_current_preprompt(self):
        """Save the current preprompt content"""
        current_name = self.preprompt_dropdown.currentText()
        if current_name == "None":
            QMessageBox.warning(
                self.parent, "No Preprompt Selected", 
                "Please select or create a preprompt first."
            )
            return
            
        content = self.preprompt_editor.toPlainText()
        
        # Validate preprompt
        if not self.preprompt_manager.validate_preprompt(content):
            QMessageBox.warning(
                self.parent, "Invalid Preprompt", 
                "The preprompt contains unbalanced brackets or other syntax issues."
            )
            return
            
        # Save preprompt
        self.preprompt_manager.add_preprompt(current_name, content)
        QMessageBox.information(
            self.parent, "Preprompt Saved", 
            f"Preprompt '{current_name}' has been saved."
        )
    
    def delete_current_preprompt(self):
        """Delete the current preprompt"""
        current_name = self.preprompt_dropdown.currentText()
        if current_name == "None":
            return
            
        confirm = QMessageBox.question(
            self.parent, "Confirm Delete", 
            f"Are you sure you want to delete the preprompt '{current_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            is_default = (current_name == self.preprompt_manager.default_preprompt)
            self.preprompt_manager.remove_preprompt(current_name)
            self.update_preprompt_dropdown()
            self.preprompt_dropdown.setCurrentIndex(0)  # Select "None"
            
            # Update default label if needed
            if is_default:
                self.update_default_label()
    
    def get_preprompt_widget(self):
        """Get the preprompt group box widget for adding to layouts"""
        return self.preprompt_group
    
    def get_current_preprompt_text(self):
        """Get the text of the currently selected preprompt and increment its usage count"""
        current_name = self.preprompt_dropdown.currentText()
        if current_name and current_name != "None":
            # Increment usage count
            self.preprompt_manager.increment_usage_count(current_name)
            return self.preprompt_manager.get_preprompt_by_name(current_name)
        return ""
    
    def validate_prompt(self, prompt_text):
        """Validate prompt text to ensure it doesn't have syntax issues"""
        return self.preprompt_manager.validate_preprompt(prompt_text)


class CollapsiblePrepromptUI:
    """UI components for collapsible preprompt functionality"""
    
    def __init__(self, parent, preprompt_manager):
        self.parent = parent
        self.preprompt_manager = preprompt_manager
        self.preprompt_ui = PrepromptUI(parent, preprompt_manager)
        self.is_expanded = False
        self.setup_ui()
        
    def setup_ui(self):
        # Create a container widget
        self.container_widget = QWidget()
        container_layout = QVBoxLayout(self.container_widget)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header bar with toggle button
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(5, 5, 5, 5)
        
        # Toggle button
        self.toggle_btn = QPushButton("➕ Preprompt")
        self.toggle_btn.setStyleSheet("text-align: left;")
        self.toggle_btn.clicked.connect(self.toggle_expansion)
        
        header_layout.addWidget(self.toggle_btn)
        header_layout.addStretch()
        
        # Add current preprompt name as label
        self.current_name_label = QLabel()
        self.update_current_name_label()
        header_layout.addWidget(self.current_name_label)
        
        # Default indicator
        self.default_indicator = QLabel()
        self.update_default_indicator()
        header_layout.addWidget(self.default_indicator)
        
        # Content widget (original preprompt UI)
        self.content_widget = self.preprompt_ui.get_preprompt_widget()
        self.content_widget.setVisible(False)
        
        # Add widgets to container
        container_layout.addWidget(header_widget)
        container_layout.addWidget(self.content_widget)
        
        # Listen for preprompt selection changes
        self.preprompt_ui.preprompt_dropdown.currentTextChanged.connect(self.update_current_name_label)
        self.preprompt_ui.preprompt_dropdown.currentTextChanged.connect(self.update_default_indicator)
    
    def toggle_expansion(self):
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        self.toggle_btn.setText("➖ Preprompt" if self.is_expanded else "➕ Preprompt")
    
    def update_current_name_label(self):
        current_text = self.preprompt_ui.preprompt_dropdown.currentText()
        if current_text and current_text != "None":
            self.current_name_label.setText(f"Selected: {current_text}")
        else:
            self.current_name_label.setText("No preprompt selected")
    
    def update_default_indicator(self):
        current_text = self.preprompt_ui.preprompt_dropdown.currentText()
        if self.preprompt_manager.use_last_as_default:
            self.default_indicator.setText("[Last as default]")
        elif self.preprompt_manager.default_preprompt:
            if current_text == self.preprompt_manager.default_preprompt:
                self.default_indicator.setText("[Default]")
            else:
                self.default_indicator.setText("")
        else:
            self.default_indicator.setText("")
    
    def get_preprompt_widget(self):
        return self.container_widget
    
    def get_current_preprompt_text(self):
        """Proxy method to get preprompt text from underlying PrepromptUI"""
        return self.preprompt_ui.get_current_preprompt_text()
    
    def validate_prompt(self, prompt_text):
        """Proxy method to validate prompt using underlying PrepromptUI"""
        return self.preprompt_ui.validate_prompt(prompt_text)