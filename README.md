# MultiProviderChat

A versatile desktop application that lets you interact with multiple AI providers (OpenAI, Anthropic Claude, and Ollama) through a unified interface. This PyQt5-based tool offers a seamless experience for using various large language models across different providers.

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

- **User-Friendly Design**
  - Simple and intuitive interface
  - Streaming responses in real-time
  - Easy configuration for multiple providers

## Dependencies

The application requires the following Python packages:

- Python 3.6+
- PyQt5
- Requests
- qtconsole (for the Jupyter console integration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Magnetron85/Chat.git
   cd Chat
   ```

2. Install the required dependencies:
   ```bash
   pip install PyQt5 requests qtconsole
   ```

## Usage

### Running the Application

cd to the folder containing pychat.py 

run
```bash
python pychat.py
```

### Configuration

1. **Setting up OpenAI**
   - Go to the "Settings" tab
   - Select "OpenAI" from the provider dropdown
   - Enter your OpenAI API key (get it from [OpenAI Platform](https://platform.openai.com/))
   - Click "Save Settings"

2. **Setting up Anthropic Claude**
   - Go to the "Settings" tab
   - Select "Claude (Anthropic)" from the provider dropdown
   - Enter your Anthropic API key (get it from [Anthropic Console](https://console.anthropic.com/))
   - Click "Save Settings"

3. **Setting up Ollama**
   - [Install Ollama](https://ollama.ai/) on your system
   - Make sure Ollama is running and listening on 0.0.0.0 
   - In the application, use the default URL (`http://localhost:11434`) or modify if Ollama is running elsewhere on the network
   - Click "Refresh Models" to load your installed Ollama models

### Using Preprompts

Preprompts let you store reusable contexts to add to your prompts in conversations (ie keep it brief):

1. Click on the "Preprompt" button to expand the preprompt panel
2. Click "New" to create a new preprompt 
3. Enter a name and content for your preprompt
4. Click "Save" to store the preprompt
5. Select your preprompt before sending messages

You can set default preprompts or configure the application to always use the last selected preprompt.

### Sending Messages

1. Select your desired provider and model
2. Type your message in the input area
3. Click "Send" or press Ctrl+Enter to submit
4. View the AI's response in the chat area

### Saving Conversations

Click "Save Chat" to export the current conversation to a text file.

## Model Management

- For Ollama, you need to [install models](https://ollama.ai/library) before they appear in the application
- For OpenAI and Anthropic, available models are loaded automatically when you have a valid API key

## Troubleshooting

- **Models not loading**: Check your API keys in the Settings tab and ensure they're valid
- **Ollama not connecting**: Make sure Ollama is running on your system
- **Error messages in responses**: Check the application log file (`ai_chat_debug.log`) for details

## License

[MIT License](LICENSE)
