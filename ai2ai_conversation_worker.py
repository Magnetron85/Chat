import logging
from PyQt5.QtCore import QThread, pyqtSignal
import time
import random
import hashlib
import json
import re


class AI2AIConversationWorker(QThread):
    """Worker thread that manages a conversation between two AI models"""
    
    # Signals
    message_added = pyqtSignal(str, str, str)  # role, content, model
    conversation_finished = pyqtSignal(bool, str)  # success, message
    progress_updated = pyqtSignal(int, int)  # current_turn, max_turns
    
    def __init__(self, db_manager, thread_id, ai1_config, ai2_config, task_prompt, max_turns, 
                 include_thinking=False, initial_context=None, custom_system_prompt=None,
                 conversation_style="standard"):
        super().__init__()
        self.db_manager = db_manager
        self.thread_id = thread_id
        self.ai1_config = ai1_config
        self.ai2_config = ai2_config
        self.task_prompt = task_prompt
        self.max_turns = max_turns
        self.stop_requested = False
        self.include_thinking = include_thinking
        self.initial_context = initial_context
        self.custom_system_prompt = custom_system_prompt
        self.conversation_style = conversation_style
        self.current_turn = 1
        self.conversation_history = []
        self.in_code_block = False
        self.code_block_buffer = ""
    
    def run(self):
        """Run the AI-to-AI conversation"""
        try:
            # Initialize conversation with task prompt
            current_prompt = self.task_prompt
            
            # Add system message for the conversation
            system_message = self._generate_system_message()
            self.db_manager.add_message(
                self.thread_id,
                "system",
                system_message
            )
            
            # Add initial context if provided
            if self.initial_context:
                self.db_manager.add_message(
                    self.thread_id,
                    "context",
                    self.initial_context
                )
                
                # Add it to the current prompt
                current_prompt = f"{self.initial_context}\n\nTask: {self.task_prompt}"
            
            # First AI starts
            current_ai = 1
            
            while self.current_turn <= self.max_turns and not self.stop_requested:
                self.progress_updated.emit(self.current_turn, self.max_turns)
                logging.debug(f"AI2AI conversation: Turn {self.current_turn}/{self.max_turns}, AI {current_ai}'s turn")
                
                if current_ai == 1:
                    # AI 1's turn
                    config = self.ai1_config
                    ai_name = f"AI 1 ({config['provider']} - {config['model']})"
                    opponent_name = f"AI 2 ({self.ai2_config['provider']} - {self.ai2_config['model']})"
                    role = "ai1"
                else:
                    # AI 2's turn
                    config = self.ai2_config
                    ai_name = f"AI 2 ({config['provider']} - {config['model']})"
                    opponent_name = f"AI 1 ({self.ai1_config['provider']} - {self.ai1_config['model']})"
                    role = "ai2"
                
                # Prepare the prompt based on conversation style
                prompt = self._prepare_prompt(config, ai_name, opponent_name, current_prompt)
                
                # Pause before making request if previous message was just added (to prevent rate limiting)
                time.sleep(0.5)
                
                # Execute the request for the appropriate provider
                try:
                    # Log that we're making the request
                    logging.debug(f"Executing AI request for {config['provider']} - {config['model']}")
                    response = self._execute_ai_request(config['provider'], config['model'], prompt)
                    
                    if not response or self.stop_requested:
                        break
                    
                    # Process the response to clean up any thinking sections
                    response = self._process_response(response, config)
                    
                    # Add the message to the database
                    self.db_manager.add_message(
                        self.thread_id,
                        role,
                        response,
                        provider=config['provider'],
                        model=config['model']
                    )
                    
                    # Add to conversation history
                    self.conversation_history.append({
                        "role": role,
                        "content": response,
                        "provider": config['provider'],
                        "model": config['model']
                    })
                    
                    # Emit signal that a message was added
                    self.message_added.emit(role, response, config['model'])
                    
                    # Update the conversation history for the next turn
                    if self.conversation_style == "detailed":
                        current_prompt += f"\n\n{ai_name}: {response}"
                    else:
                        current_prompt += f"\n\n{role.upper()}: {response}"
                    
                except Exception as e:
                    logging.error(f"Error executing AI request: {str(e)}")
                    self.conversation_finished.emit(False, f"Error on turn {self.current_turn}: {str(e)}")
                    return
                
                # Switch to the other AI
                current_ai = 2 if current_ai == 1 else 1
                
                # Increment turn if both AIs have spoken
                if current_ai == 1:
                    self.current_turn += 1
                
                # Small delay to avoid hammering the API
                delay = random.uniform(1.0, 2.0)  # Randomize delay to be more natural
                time.sleep(delay)
            
            # Conversation complete
            if self.stop_requested:
                self.conversation_finished.emit(True, "AI-to-AI conversation stopped by user.")
            else:
                self.conversation_finished.emit(True, "AI-to-AI conversation completed successfully.")
            
        except Exception as e:
            logging.error(f"Error in AI-to-AI conversation: {str(e)}")
            self.conversation_finished.emit(False, f"Error: {str(e)}")
    
    def _generate_system_message(self):
        """Generate the system message for the conversation"""
        if self.custom_system_prompt:
            # Use custom system prompt if provided
            base_message = self.custom_system_prompt
        else:
            # Generate default system message
            base_message = (
                f"AI-to-AI Conversation\n\n"
                f"Task: {self.task_prompt}\n\n"
                f"AI 1: {self.ai1_config['provider']} - {self.ai1_config['model']}\n"
                f"AI 2: {self.ai2_config['provider']} - {self.ai2_config['model']}\n"
                f"Maximum Turns: {self.max_turns}"
            )
            
            # Add conversation style description
            if self.conversation_style == "debate":
                base_message += "\n\nConversation Style: Debate - The AIs will discuss opposing viewpoints on the topic."
            elif self.conversation_style == "collaborative":
                base_message += "\n\nConversation Style: Collaborative - The AIs will work together to solve the problem."
            elif self.conversation_style == "detailed":
                base_message += "\n\nConversation Style: Detailed - The AIs will include their identities in each message."
                
        return base_message
    
    def _prepare_prompt(self, config, ai_name, opponent_name, current_prompt):
        """Prepare the prompt for the AI, incorporating conversation style and history"""
        # Get system prompt from config if available
        system_prompt = config.get('prompt', "")
        
        # Adjust the prompt based on conversation style
        if self.conversation_style == "debate":
            role_instruction = f"You are {ai_name} having a debate with {opponent_name} about this topic. Present compelling arguments and respond to your opponent's points."
        elif self.conversation_style == "collaborative":
            role_instruction = f"You are {ai_name} collaborating with {opponent_name} to solve this problem together. Build upon each other's ideas."
        elif self.conversation_style == "detailed":
            role_instruction = f"You are {ai_name} having a conversation with {opponent_name}. Always include your perspective and reasoning."
        else:  # standard
            role_instruction = f"You are {ai_name} having a conversation with {opponent_name} about this task."
        
        # Create the prompt with appropriate structure
        if system_prompt:
            full_prompt = (
                f"{system_prompt}\n\n"
                f"{role_instruction}\n"
                f"Task: {self.task_prompt}\n\n"
                f"Current conversation:\n{current_prompt}\n\n"
                f"Your response (as {ai_name}):"
            )
        else:
            full_prompt = (
                f"{role_instruction}\n"
                f"Task: {self.task_prompt}\n\n"
                f"Current conversation:\n{current_prompt}\n\n"
                f"Your response (as {ai_name}):"
            )
            
        return full_prompt
    
    def _process_response(self, response, config):
        """Process the AI response to clean up any artifacts or format issues"""
        if not response:
            return ""
            
        # Remove thinking sections if not included
        if not self.include_thinking:
            # Remove Deepseek thinking sections
            response = self._remove_thinking_sections(response, "<thinking>", "</thinking>")
            # Remove standard Ollama thinking sections
            response = self._remove_thinking_sections(response, "<think>", "</think>")
            
        # Handle provider-specific formatting issues
        provider = config.get('provider', '').lower()
        
        # Clean up response prefixes that some models might add
        prefixes_to_remove = [
            "Here's my response:", 
            "As requested, here is my response:", 
            "My response as AI", 
            "AI 1:", 
            "AI 2:",
            "AI1:", 
            "AI2:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                
        # Remove any instances where the AI signs its name at the end
        signature_patterns = [
            r'\n\s*-- AI 1\s*$',
            r'\n\s*-- AI 2\s*$',
            r'\n\s*- AI 1\s*$',
            r'\n\s*- AI 2\s*$',
            r'\n\s*AI 1\s*$',
            r'\n\s*AI 2\s*$'
        ]
        
        for pattern in signature_patterns:
            response = re.sub(pattern, '', response)
            
        # Format code blocks properly
        formatted_response = self._format_code_blocks(response)
            
        return formatted_response.strip()

    # Add this new method to format code blocks:
    def _format_code_blocks(self, text):
        """Format code blocks properly, similar to how pychat.py handles them"""
        if "```" not in text:
            return text
            
        formatted_parts = []
        self.in_code_block = False
        self.code_block_buffer = ""
        
        # Split by code blocks
        parts = re.split(r'(```(?:\w*)\n[\s\S]*?\n```)', text)
        
        for part in parts:
            if part.strip() and part.startswith("```") and part.endswith("```"):
                # This is a code block - we want to preserve it intact
                formatted_parts.append(part)
            elif part.strip():
                # Normal text part
                formatted_parts.append(part)
        
        # Join all the processed parts
        return "".join(formatted_parts)
    
    def _remove_thinking_sections(self, text, start_tag, end_tag):
        """Remove sections enclosed in specific tags from text"""
        if not text:
            return text
            
        result = text
        start_idx = result.find(start_tag)
        
        while start_idx != -1:
            end_idx = result.find(end_tag, start_idx)
            if end_idx != -1:
                # Remove the section including tags
                result = result[:start_idx] + result[end_idx + len(end_tag):]
                # Look for next section
                start_idx = result.find(start_tag)
            else:
                # End tag not found, break to avoid infinite loop
                break
                
        return result
    
    def _execute_ai_request(self, provider_id, model, prompt):
        """Execute a request to an AI provider and return the response"""
        try:
            # Import the necessary handlers based on provider type
            from openai_handler import OpenAIRequestWorker
            from anthropic_handler import AnthropicRequestWorker
            from ollama_handler import OllamaRequestWorker
            
            # Get the providers config and API keys from the main app
            from pychat import PROVIDERS, RequestWorker
            import sys
            from PyQt5.QtWidgets import QApplication
            
            # Variable to hold the response
            response_text = None
            
            # Get API key from the main application window
            api_key = None
            for widget in QApplication.topLevelWidgets():
                if hasattr(widget, 'current_api_keys'):
                    api_key = widget.current_api_keys.get(provider_id)
                    logging.debug(f"Found API key for {provider_id} in main window")
                    break
            
            # Log API key status
            logging.debug(f"Using API key for {provider_id}: {'Found' if api_key else 'Not found'}")
            
            # Create and execute the appropriate worker based on provider
            if provider_id.lower() == "openai":
                # Create synchronous worker
                worker = OpenAIRequestWorker(
                    PROVIDERS["openai"]["api_url"],
                    model,
                    prompt,
                    api_key,
                    stream=False  # No streaming for AI-to-AI conversations
                )
                
                # Execute synchronously
                worker.finished.connect(lambda text, success: setattr(self, '_response', text if success else f"Error: {text}"))
                worker.run()  # Run directly without start() to make it synchronous
                response_text = getattr(self, '_response', None)
                
            elif provider_id.lower() == "anthropic":
                # Create synchronous worker
                worker = AnthropicRequestWorker(
                    PROVIDERS["anthropic"]["api_url"],
                    model,
                    prompt,
                    api_key,
                    stream=False  # No streaming for AI-to-AI conversations
                )
                
                # Execute synchronously
                worker.finished.connect(lambda text, success: setattr(self, '_response', text if success else f"Error: {text}"))
                worker.run()  # Run directly without start() to make it synchronous
                response_text = getattr(self, '_response', None)
                
            elif provider_id.lower() == "ollama":
                # Get base URL from the config
                base_url = PROVIDERS["ollama"]["api_url"]
                
                # Create synchronous worker
                worker = OllamaRequestWorker(
                    base_url,
                    model,
                    prompt,
                    stream=False  # No streaming for AI-to-AI conversations
                )
                
                # Execute synchronously
                worker.finished.connect(lambda text, success: setattr(self, '_response', text if success else f"Error: {text}"))
                worker.run()  # Run directly without start() to make it synchronous
                response_text = getattr(self, '_response', None)
                
            else:
                # Use generic request worker for other providers
                provider_config = PROVIDERS.get(provider_id)
                if not provider_config:
                    return f"Unknown provider: {provider_id}"
                    
                # Create the worker
                worker = RequestWorker(
                    provider_config,
                    model,
                    prompt,
                    api_key=api_key,
                    stream=False  # No streaming for AI-to-AI conversations
                )
                
                # Execute synchronously
                worker.finished.connect(lambda text, success: setattr(self, '_response', text if success else f"Error: {text}"))
                worker.run()  # Run directly without start() to make it synchronous
                response_text = getattr(self, '_response', None)
                
            # Return the response text or a default message
            if response_text:
                return response_text
            else:
                return f"No response received from {provider_id} API"
                
        except Exception as e:
            logging.error(f"Error executing AI request: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def stop(self):
        """Stop the conversation"""
        logging.debug("AI2AI conversation stop requested")
        self.stop_requested = True
        
    def get_conversation_summary(self):
        """Return a summary of the conversation"""
        if not self.conversation_history:
            return "No conversation has taken place yet."
            
        ai1_messages = [msg for msg in self.conversation_history if msg["role"] == "ai1"]
        ai2_messages = [msg for msg in self.conversation_history if msg["role"] == "ai2"]
        
        summary = {
            "total_turns": self.current_turn - 1,  # Subtract 1 because current_turn is the next turn
            "total_messages": len(self.conversation_history),
            "ai1_messages": len(ai1_messages),
            "ai2_messages": len(ai2_messages),
            "ai1_config": {
                "provider": self.ai1_config["provider"],
                "model": self.ai1_config["model"]
            },
            "ai2_config": {
                "provider": self.ai2_config["provider"],
                "model": self.ai2_config["model"]
            },
            "task": self.task_prompt,
            "conversation_style": self.conversation_style
        }
        
        return summary