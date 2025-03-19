import re
from PyQt5.QtWidgets import QTextEdit, QApplication
from PyQt5.QtCore import QMimeData, Qt, QRect
from PyQt5.QtGui import (QFont, QTextCursor, QColor, QTextCharFormat, QSyntaxHighlighter, 
                        QTextBlockFormat, QPainter, QTextFormat, QBrush, QPen, QTextOption)

class CodeHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for code blocks"""
    
    def __init__(self, parent=None, language="python"):
        super().__init__(parent)
        self.language = language.lower()
        self.highlighting_rules = []
        
        if self.language in ["python", "py"]:
            self._setup_python_rules()
        elif self.language in ["javascript", "js"]:
            self._setup_javascript_rules()
        else:
            # Default rules for other languages
            self._setup_generic_rules()
            

    
    def _setup_python_rules(self):
        # Python keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#0000FF"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "and", "as", "assert", "break", "class", "continue", "def",
            "del", "elif", "else", "except", "False", "finally", "for",
            "from", "global", "if", "import", "in", "is", "lambda", "None",
            "not", "or", "pass", "raise", "return", "True", "try", "while",
            "with", "yield"
        ]
        
        for word in keywords:
            pattern = r'\b' + word + r'\b'
            self.highlighting_rules.append((re.compile(pattern), keyword_format))
        
        # String literals
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000"))
        self.highlighting_rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), string_format))
        self.highlighting_rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))
        
        # Function calls
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#800080"))
        self.highlighting_rules.append((re.compile(r'\b[A-Za-z0-9_]+(?=\()'), function_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF8000"))
        self.highlighting_rules.append((re.compile(r'\b\d+\b'), number_format))
        
        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#808080"))
        self.highlighting_rules.append((re.compile(r'#[^\n]*'), comment_format))
    
    def _setup_javascript_rules(self):
        # JavaScript keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#0000FF"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "break", "case", "catch", "class", "const", "continue", "debugger",
            "default", "delete", "do", "else", "export", "extends", "false",
            "finally", "for", "function", "if", "import", "in", "instanceof",
            "new", "null", "return", "super", "switch", "this", "throw", "true",
            "try", "typeof", "var", "void", "while", "with", "yield", "let"
        ]
        
        for word in keywords:
            pattern = r'\b' + word + r'\b'
            self.highlighting_rules.append((re.compile(pattern), keyword_format))
        
        # String literals
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000"))
        self.highlighting_rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), string_format))
        self.highlighting_rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))
        self.highlighting_rules.append((re.compile(r"`[^`\\]*(\\.[^`\\]*)*`"), string_format))
        
        # Function calls
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#800080"))
        self.highlighting_rules.append((re.compile(r'\b[A-Za-z0-9_]+(?=\()'), function_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF8000"))
        self.highlighting_rules.append((re.compile(r'\b\d+\b'), number_format))
        
        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#808080"))
        self.highlighting_rules.append((re.compile(r'//[^\n]*'), comment_format))
        self.highlighting_rules.append((re.compile(r'/\*[\s\S]*?\*/', re.MULTILINE), comment_format))
    
    def _setup_generic_rules(self):
        # Generic patterns for most languages
        # String literals
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000"))
        self.highlighting_rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), string_format))
        self.highlighting_rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF8000"))
        self.highlighting_rules.append((re.compile(r'\b\d+\b'), number_format))
    
    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), format)


class EnhancedChatBrowser(QTextEdit):
    """Custom QTextEdit with beautiful code block formatting and copy capability"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Segoe UI", 10))
        
        # Configure document margin and line spacing
        self.document().setDocumentMargin(15)
        
        # Set up options - FIXED: removed setLineHeight which doesn't exist
        option = self.document().defaultTextOption()
        self.document().setDefaultTextOption(option)
        
        # Set line spacing using text block format
        block_format = QTextBlockFormat()
        block_format.setLineHeight(150, QTextBlockFormat.ProportionalHeight)
        
        # Apply the format to the current block
        cursor = self.textCursor()
        cursor.setBlockFormat(block_format)
        self.setTextCursor(cursor)
        
        # State tracking for code blocks
        self.in_code_block = False
        self.code_block_buffer = ""
        self.code_language = ""
        
        # Track if we're in a streaming response
        self.is_streaming = False
        self.streaming_text = ""
        
        # Add a dictionary to track streaming responses by unique IDs
        self.streaming_responses = {}
        self.current_streaming_id = None
        
        # Style the widget
        self.setStyleSheet("""
            QTextEdit {
                background-color: white;
                color: #24292e;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
                selection-background-color: #b3d7ff;
                padding: 5px;
            }
        """)
    
    def clear(self):
        """Clear the chat display"""
        super().clear()
    
    def append(self, text):
        """Add text to the chat display with proper formatting"""
        # Reset streaming state for a complete message
        self.is_streaming = False
        self.streaming_text = ""
        
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Format based on message type
        if text.startswith("[SYSTEM]"):
            # System message in italic gray
            format = QTextCharFormat()
            format.setForeground(QColor("#6a737d"))
            format.setFontItalic(True)
            cursor.insertText(text + "\n", format)
            
        elif text.startswith(">"):
            # User message in bold
            format = QTextCharFormat()
            format.setFontWeight(QFont.Bold)
            cursor.insertText(text + "\n", format)
            
        elif not text.strip():
            # Empty line
            cursor.insertBlock()
            
        else:
            # AI response - process for code blocks
            if "```" in text:
                # Split by code blocks
                pattern = r'(```(?:\w*)\n[\s\S]*?\n```)'
                parts = re.split(pattern, text)
                
                for part in parts:
                    if part.strip() and part.startswith("```") and part.endswith("```"):
                        self._insert_code_block(cursor, part)
                    elif part.strip():
                        cursor.insertText(part)
                
                # Add final newline
                cursor.insertBlock()
            else:
                # Normal text
                format = QTextCharFormat()
                format.setForeground(QColor("#24292e"))
                cursor.insertText(text + "\n", format)
        
        # Update cursor and scroll to bottom
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
    
    def begin_streaming_response(self):
        """Start a new streaming response session"""
        # Generate a unique ID for this response
        import uuid
        self.current_streaming_id = str(uuid.uuid4())
        
        # Create a placeholder for this response
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Initialize empty response in dictionary
        self.streaming_responses[self.current_streaming_id] = ""
        
        # Return the streaming ID
        return self.current_streaming_id

    def append_streaming_chunk(self, streaming_id, chunk_text):
        """Add a chunk to an existing streaming response"""
        if streaming_id not in self.streaming_responses:
            return False
        
        # Add to our internal buffer
        self.streaming_responses[streaming_id] += chunk_text
        
        # Replace the entire response text
        self.update_streaming_response(streaming_id)
        
        return True

    def update_streaming_response(self, streaming_id):
        """Update the display with the current state of the streaming response"""
        if streaming_id not in self.streaming_responses:
            return False
        
        current_text = self.streaming_responses[streaming_id]
        
        # Process the text for code blocks before displaying
        formatted_text = current_text
        
        # Check if there are complete code blocks in the text
        if "```" in formatted_text:
            # Find all complete code blocks
            blocks = []
            in_block = False
            start_pos = 0
            
            for i, char in enumerate(formatted_text):
                if i+2 < len(formatted_text) and formatted_text[i:i+3] == "```":
                    if not in_block:  # Start of a code block
                        in_block = True
                        start_pos = i
                    else:  # End of a code block
                        in_block = False
                        blocks.append((start_pos, i+3))
            
            # Process complete code blocks and replace them with placeholders
            processed_blocks = []
            offset = 0
            
            for start, end in blocks:
                # Adjust positions for previous replacements
                adj_start = start - offset
                adj_end = end - offset
                
                # Extract the code block
                code_block = formatted_text[adj_start:adj_end]
                
                # Process the code block to get formatted version
                processed_block = f"__CODE_BLOCK_{len(processed_blocks)}__"
                processed_blocks.append((code_block, processed_block))
                
                # Replace in the formatted text
                formatted_text = formatted_text[:adj_start] + processed_block + formatted_text[adj_end:]
                
                # Update offset
                offset += (end - start) - len(processed_block)
        
        # Get the document and create a cursor
        cursor = QTextCursor(self.document())
        
        # Find the position where we need to start replacing content
        # Look for the most recent system message 
        doc_text = self.toPlainText()
        start_pos = doc_text.rfind("[SYSTEM] Processing request...")
        
        if start_pos >= 0:
            # Position cursor after this system message and its newline
            cursor.setPosition(start_pos)
            cursor.movePosition(QTextCursor.EndOfLine)
            cursor.movePosition(QTextCursor.Right, QTextCursor.MoveAnchor, 1)  # Move past the newline
            
            # Select all text from this point to the end
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            
            # Remove existing content
            cursor.removeSelectedText()
            
            # Insert the processed text with code block handling
            if "```" in current_text and processed_blocks:
                # Insert text with properly formatted code blocks
                parts = formatted_text.split("__CODE_BLOCK_")
                
                # Insert the first part
                format = QTextCharFormat()
                format.setForeground(QColor("#24292e"))
                cursor.insertText(parts[0], format)
                
                # Insert each code block followed by the text that comes after it
                for i in range(1, len(parts)):
                    # Extract the code block index and remaining text
                    if "_" in parts[i]:
                        block_idx_str, remaining = parts[i].split("__", 1)
                        try:
                            block_idx = int(block_idx_str)
                            if block_idx < len(processed_blocks):
                                # Insert the code block
                                self._insert_code_block(cursor, processed_blocks[block_idx][0])
                                
                                # Insert the remaining text
                                cursor.insertText(remaining, format)
                        except ValueError:
                            # In case of parsing error, just insert as text
                            cursor.insertText(parts[i], format)
                    else:
                        # Fallback for parsing errors
                        cursor.insertText(parts[i], format)
            else:
                # Simple case: no code blocks
                format = QTextCharFormat()
                format.setForeground(QColor("#24292e"))
                cursor.insertText(formatted_text, format)
        else:
            # If we can't find the system message, just append at the end
            cursor.movePosition(QTextCursor.End)
            format = QTextCharFormat()
            format.setForeground(QColor("#24292e"))
            cursor.insertText(formatted_text, format)
        
        # Update cursor and ensure it's visible
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        
        return True

    def end_streaming_response(self, streaming_id):
        """Finalize a streaming response"""
        if streaming_id not in self.streaming_responses:
            return False
        
        # Just clean up the tracking
        self.streaming_responses.pop(streaming_id)
        if self.current_streaming_id == streaming_id:
            self.current_streaming_id = None
        
        return True
    
    def insertHtml(self, html):
        """Handle streaming content with proper text flow"""
        cursor = self.textCursor()
        
        # Get current position and text
        cursor.movePosition(QTextCursor.End)
        
        # For code blocks, use the existing logic
        if "```" in html:
            # (existing code block handling)
            return
        
        # KEY FIX: For normal text, don't create new blocks or paragraphs
        # Strip HTML tags for safety
        plain_text = re.sub(r'<[^>]*>', '', html)
        
        # Find the current paragraph and text
        cursor.movePosition(QTextCursor.StartOfBlock)
        has_text = not cursor.atEnd()
        
        # Format for text insertion
        format = QTextCharFormat()
        format.setForeground(QColor("#24292e"))
        
        # If we're continuing a paragraph, we want to append in-place
        # rather than creating a new line or paragraph
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(plain_text, format)
        
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        
        # If we have a pending code block, keep accumulating
        if self.in_code_block:
            self.code_block_buffer += html
            return
        
        # For normal text in streaming mode, accumulate and insert
        # Strip HTML tags for safety
        plain_text = re.sub(r'<[^>]*>', '', html)
        self.streaming_text += plain_text
        
        # Update the display with the current accumulated text
        # Find the start of our streaming text
        if cursor.block().text() == "":
            # If we're at a blank line, start a new block
            format = QTextCharFormat()
            format.setForeground(QColor("#24292e"))
            cursor.insertText(self.streaming_text, format)
        else:
            # Otherwise, replace the current line with our accumulated text
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            format = QTextCharFormat()
            format.setForeground(QColor("#24292e"))
            cursor.insertText(self.streaming_text, format)
        
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
    
    def setHtml(self, html):
        """Set the entire HTML content after cleaning it"""
        # Strip HTML tags and set as plain text
        plain_text = re.sub(r'<[^>]*>', '', html)
        self.setPlainText(plain_text)
    
    def _insert_code_block(self, cursor, code_block_text):
        """Insert a formatted code block at the cursor position"""
        # Extract language and code
        if "\n" in code_block_text:
            first_line = code_block_text.split("\n", 1)[0]
            language = first_line.replace("```", "").strip() or "python"
            code = code_block_text.split("\n", 1)[1].rsplit("```", 1)[0]
        else:
            language = "python"
            code = code_block_text.replace("```", "").strip()
        
        # Insert a divider before the code block
        cursor.insertBlock()
        
        # Create a background block format for the code
        block_format = QTextBlockFormat()
        block_format.setBackground(QColor("#F6F8FA"))
        block_format.setLeftMargin(10)
        block_format.setRightMargin(10)
        block_format.setTopMargin(5)
        block_format.setBottomMargin(5)
        cursor.setBlockFormat(block_format)
        
        # Insert the language label
        lang_format = QTextCharFormat()
        lang_format.setForeground(QColor("#6A737D"))
        lang_format.setFontWeight(QFont.Bold)
        lang_format.setFontFamily("Segoe UI")
        cursor.insertText(f"[{language}]", lang_format)
        cursor.insertBlock()
        
        # Insert the code with monospace font
        code_format = QTextCharFormat()
        code_format.setFontFamily("Consolas, Liberation Mono, Menlo, monospace")
        code_format.setFontPointSize(9)
        
        # We'll highlight each line according to the language
        highlighter = CodeHighlighter(None, language)
        
        # Insert code line by line with highlighting
        for line in code.split('\n'):
            # Apply syntax highlighting
            highlighted_format = QTextCharFormat(code_format)
            if line.strip():
                # Create a temporary document to highlight the line
                for rule_pattern, rule_format in highlighter.highlighting_rules:
                    for match in rule_pattern.finditer(line):
                        # Create a text cursor at the match position
                        match_start = match.start()
                        match_length = match.end() - match.start()
                        
                        # Apply the format and insert the highlighted text
                        temp_format = QTextCharFormat(code_format)
                        temp_format.setForeground(rule_format.foreground())
                        if rule_format.fontWeight() > QFont.Normal:
                            temp_format.setFontWeight(rule_format.fontWeight())
                        
                        # Insert text up to the match
                        if match_start > 0:
                            cursor.insertText(line[:match_start], code_format)
                        
                        # Insert the highlighted match
                        cursor.insertText(line[match_start:match_start+match_length], temp_format)
                        
                        # Update the line to continue after the match
                        line = line[match_start+match_length:]
                        break
                    if not line:  # If we've processed the whole line
                        break
                
                # Insert any remaining part of the line
                if line:
                    cursor.insertText(line, code_format)
            else:
                cursor.insertText(line, code_format)
            
            cursor.insertBlock()
            cursor.setBlockFormat(block_format)
        
        # Reset the block format
        normal_block = QTextBlockFormat()
        cursor.setBlockFormat(normal_block)
        
        # Add copy button
        copy_format = QTextCharFormat()
        copy_format.setForeground(QColor("#0366d6"))
        copy_format.setAnchor(True)
        # Store the code to copy in the URL with a special prefix
        copy_format.setAnchorHref(f"copy:{code}")
        cursor.insertText("[Copy Code]", copy_format)
        cursor.insertBlock()
    
    def mousePressEvent(self, event):
        """Track mouse press events for copy link clicks"""
        if event.button() == Qt.LeftButton:
            cursor = self.cursorForPosition(event.pos())
            char_format = cursor.charFormat()
            if char_format.isAnchor():
                url = char_format.anchorHref()
                if url.startswith("copy:"):
                    # Extract the code to copy
                    code = url[5:]
                    
                    # Copy to clipboard
                    clipboard = QApplication.clipboard()
                    mime_data = QMimeData()
                    mime_data.setText(code)
                    clipboard.setMimeData(mime_data)
                    
                    # Add a status message
                    self.append("\n[SYSTEM] Code copied to clipboard")
                    
                    # Prevent default handling
                    event.accept()
                    return
        
        super().mousePressEvent(event)
    
    def set_dark_mode(self, enabled):
        """Toggle between dark and light mode"""
        if enabled:
            # Dark mode
            self.setStyleSheet("""
                QTextEdit {
                    background-color: #0d1117;
                    color: #e1e4e8;
                    border: 1px solid #30363d;
                    border-radius: 6px;
                    selection-background-color: #3b4a63;
                    padding: 5px;
                }
            """)
        else:
            # Light mode
            self.setStyleSheet("""
                QTextEdit {
                    background-color: white;
                    color: #24292e;
                    border: 1px solid #e1e4e8;
                    border-radius: 6px;
                    selection-background-color: #b3d7ff;
                    padding: 5px;
                }
            """)