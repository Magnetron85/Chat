import re
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.QtCore import QMimeData
from PyQt5.QtGui import QFont

# Import qtconsole components
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

class EnhancedTextBrowser(QWidget):
    """Enhanced chat browser with code formatting using qtconsole"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the Jupyter console widget
        self.jupyter_widget = RichJupyterWidget(self)
        self.jupyter_widget.setStyleSheet("background-color: white;")
        self.jupyter_widget.font = QFont("Segoe UI", 10)
        
        # Set up the kernel
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()
        
        # Connect the widget to the kernel
        self.jupyter_widget.kernel_manager = self.kernel_manager
        self.jupyter_widget.kernel_client = self.kernel_client
        
        # Configure the console display
        self.jupyter_widget.set_default_style("linux")  # Clean black-on-white style
        
        # Hide input area, we don't want users typing in this widget
        self.jupyter_widget.hide_input()
        
        # Add widget to layout
        self.layout.addWidget(self.jupyter_widget)
        
        # Initialize state variables
        self.in_code_block = False
        self.code_block_buffer = ""
        self.code_language = ""
        
    def clear(self):
        """Clear the chat display"""
        self.jupyter_widget.reset(clear=True)
    
    def append(self, text):
        """Add text to the chat display with proper formatting"""
        if text.startswith("[SYSTEM]"):
            # System message in gray
            html = f'<div style="color: gray; font-style: italic;">{text}</div>'
            self.jupyter_widget._append_html(html)
        
        elif text.startswith(">"):
            # User message in bold
            html = f'<div style="font-weight: bold;">{text}</div>'
            self.jupyter_widget._append_html(html)
        
        elif not text.strip():
            # Empty line
            self.jupyter_widget._append_html("<br>")
        
        else:
            # AI response - handle markdown and code blocks
            if "```" in text:
                # Process code blocks
                parts = re.split(r'(```(?:\w*)\n[\s\S]*?\n```)', text)
                for part in parts:
                    if part.startswith("```") and part.endswith("```"):
                        # This is a code block
                        self._process_code_block(part)
                    else:
                        # Regular text
                        self.jupyter_widget._append_html(f'<div>{part}</div>')
            else:
                # No code blocks, just append as normal text
                self.jupyter_widget._append_html(f'<div>{text}</div>')
    
    def _process_code_block(self, code_block):
        """Process and display a code block with syntax highlighting"""
        # Extract language and code
        if "\n" in code_block:
            first_line = code_block.split("\n", 1)[0]
            language = first_line.replace("```", "").strip() or "python"  # Default to Python
            code = code_block.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        else:
            language = "python"
            code = code_block.replace("```", "").strip()
        
        # Use the kernel to execute the code (in display-only mode)
        if language.lower() in ["python", "py"]:
            # For Python, use the actual kernel with %%capture to prevent execution
            cmd = f"from IPython.display import display, Markdown, Code\ndisplay(Code('''{code}''', language='{language}'))"
            self.kernel.shell.run_cell(cmd)
        else:
            # For other languages, use Markdown code block format
            cmd = f"from IPython.display import display, Markdown\ndisplay(Markdown('```{language}\\n{code}\\n```'))"
            self.kernel.shell.run_cell(cmd)
        
        # Add copy button
        copy_html = f'<div><a href="#" onclick="navigator.clipboard.writeText(`{code}`);return false;" style="color:blue;text-decoration:underline;">[Copy Code]</a></div>'
        self.jupyter_widget._append_html(copy_html)
    
    def insertHtml(self, html):
        """Handle streaming content, particularly code blocks"""
        # Check for code block markers
        if "```" in html:
            # Handle code blocks in streaming content
            if self.in_code_block:
                # We're inside a code block, keep accumulating
                self.code_block_buffer += html
                
                # Check if this chunk completes the code block
                if "```" in html and html.count("```") % 2 == 1:
                    # Process the complete code block
                    self._process_code_block(self.code_block_buffer)
                    self.in_code_block = False
                    self.code_block_buffer = ""
                return
            
            # Check if this starts a new code block
            if "```" in html and html.count("```") % 2 == 1:
                # This starts a new code block
                self.in_code_block = True
                self.code_block_buffer = html
                return
        
        # If we have a pending code block, keep accumulating
        if self.in_code_block:
            self.code_block_buffer += html
            return
        
        # For regular text, just insert as HTML
        # First strip any HTML tags for safety
        clean_text = re.sub(r'<[^>]*>', '', html)
        self.jupyter_widget._append_html(f"{clean_text}")
    
    def setHtml(self, html):
        """Set the entire HTML content - clear first then append"""
        self.clear()
        self.jupyter_widget._append_html(html)
    
    def toPlainText(self):
        """Get plain text content"""
        # This is a simplified implementation
        return self.jupyter_widget._control.toPlainText()
    
    def set_dark_mode(self, enabled):
        """Toggle between dark and light mode"""
        if enabled:
            # Dark mode
            self.jupyter_widget.set_default_style("monokai")
            self.jupyter_widget.setStyleSheet("background-color: #0d1117; color: #e1e4e8;")
        else:
            # Light mode
            self.jupyter_widget.set_default_style("linux")
            self.jupyter_widget.setStyleSheet("background-color: white; color: #24292e;")
    
    def verticalScrollBar(self):
        """Access the scroll bar for compatibility with previous code"""
        return self.jupyter_widget._control.verticalScrollBar()
    
    def textCursor(self):
        """Get text cursor for compatibility"""
        return self.jupyter_widget._control.textCursor()
    
    def setTextCursor(self, cursor):
        """Set text cursor for compatibility"""
        self.jupyter_widget._control.setTextCursor(cursor)


# Update the MultiProviderChat class to use the new component
# Replace instances of EnhancedTextBrowser with EnhancedChatBrowser

# Note: You'll need to install qtconsole with:
# pip install qtconsole