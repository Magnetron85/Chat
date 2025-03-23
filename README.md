# Pychat
A versatile desktop application that lets you interact with multiple AI providers (OpenAI, Anthropic Claude, and Ollama) through a unified interface. This PyQt5-based tool offers a seamless experience for using various large language models across different providers.

![Screenshot of the application](screenshot.png)

## Features
- **Multiple AI Provider Support**
  - OpenAI (GPT-4, GPT-3.5, etc.)
  - Anthropic Claude (Claude 3 Opus, Sonnet, Haiku, etc.)
  - Ollama (for local open-source models)
- **Rich Text Interface**
  - Syntax highlighting for code blocks
  - Streaming support for real-time responses
  - Copy code button for easy code reuse
- **Advanced Features**
  - Preprompt system for reusable context templates
  - "Show thinking" option for DeepSeek
  - Save and load conversations
  - Customizable API endpoints
  - Thread-based conversation management
  - AI-to-AI conversation feature
- **User-Friendly Design**
  - Simple and intuitive interface
  - Streaming responses in real-time
  - Easy configuration for multiple providers

## Important User Information

### Data Storage & Privacy
- **Local Storage**: All conversations are stored locally in an SQLite database (`chat_history.db`)
- **API Keys**: Your API keys are stored locally using Qt's settings mechanism and are not transmitted beyond the respective API services
- **Logging**: The application logs activities to `ai_chat_debug.log`, which may include message content for debugging purposes
- **No Data Sharing**: Pychat does not send your conversations to any servers except the AI provider APIs you configure

### Security Considerations
- **Credentials**: API keys are stored locally in Qt's settings storage; ensure your computer is secure
- **API Usage**: Your API usage with OpenAI and Anthropic will incur costs according to those services' pricing
- **Network Access**: The application requires internet access to communicate with remote APIs
- **Local Models**: Using Ollama allows you to run models locally with no data sent to external services

## Dependencies
The application requires the following Python packages:
- Python 3.6+
- PyQt5
- Requests
- qtconsole (for the Jupyter console integration)
- fuzzywuzzy (for search functionality)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Magnetron85/PyChat.git
   ```
2. Install the required dependencies: 
   ```bash
   pip install PyQt5 requests qtconsole fuzzywuzzy
   ```
3. Running the application: 
   ```bash
   python pychat.py
   ```


#### Setting up OpenAI 
- Go to the "Settings" tab
- Select "OpenAI" from the provider dropdown
- Enter your OpenAI API key (get it from OpenAI Platform)
- Click "Save Settings"

#### Setting up Anthropic Claude 
- Go to the "Settings" tab
- Select "Claude (Anthropic)" from the provider dropdown
- Enter your Anthropic API key (get it from Anthropic Console)
- Click "Save Settings"

#### Setting up Ollama 
- Install Ollama on your system
- Make sure Ollama is running and listening on 0.0.0.0
- In the application, use the default URL (http://localhost:11434) or modify if Ollama is running elsewhere on the network
- Click "Refresh Models" to load your installed Ollama models

### Using Preprompts
Preprompts let you store reusable contexts to add to your prompts in conversations (ie keep it brief):
- Click on the "Preprompt" button to expand the preprompt panel
- Click "New" to create a new preprompt
- Enter a name and content for your preprompt
- Click "Save" to store the preprompt
- Select your preprompt before sending messages 

You can set default preprompts or configure the application to always use the last selected preprompt.

### Sending Messages
- Select your desired provider and model
- Type your message in the input area
- Click "Send" or press Ctrl+Enter to submit
- View the AI's response in the chat area

### Conversation Management
- Conversations are organized into threads that are stored locally
- Use "File > New Thread" to start a new conversation
- The thread list on the left shows all your conversations
- Right-click on threads for options like rename, archive, or delete

### Saving Conversations
- Click "Save Chat" to export the current conversation to a text file
- Use "File > Export Thread" to save a conversation in JSON format
- Use "File > Import Thread" to load previously exported conversations

### Model Management
- For Ollama, you need to install models before they appear in the application
- For OpenAI and Anthropic, available models are loaded automatically when you have a valid API key

## Troubleshooting
- **Models not loading**: Check your API keys in the Settings tab and ensure they're valid
- **Ollama not connecting**: Make sure Ollama is running on your system
- **Error messages in responses**: Check the application log file (ai_chat_debug.log) for details
- **API Key Issues**: If you receive authorization errors, verify your API keys are correct and have not expired
- **Database Issues**: If experiencing data loss, check file permissions for the chat_history.db file

## Data Cleanup
If you want to remove all stored data:
- Delete the chat_history.db file to remove all conversations
- API keys can be cleared through the Settings tab for each provider
- Delete the ai_chat_debug.log file to remove all debug logs

## License
MIT License
