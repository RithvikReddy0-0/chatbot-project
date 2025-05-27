# 🤖 Chatbot Project

This is a simple conversational chatbot built using the [Facebook BlenderBot](https://huggingface.co/facebook/blenderbot-400M-distill) model via Hugging Face Transformers. It runs locally in a terminal and remembers the context of the conversation.

---

## Features

- Uses the BlenderBot 400M distilled model for conversational AI
- Maintains conversation history for better context
- Terminal-based chatbot for quick local testing
- Flask web server providing an interactive chat UI
- Simple, modular, and extensible design

---

## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/RithvikReddy0-0/chatbot-project.git
    cd chatbot-project

2. **Install dependencies**

    ```bash
    pip install transformers torch flask flask-cors


---

## Usage

Run the chatbot server

    ```bash
    python app.py

This will start the Flask web server locally (default at http://127.0.0.1:5000). Open your browser and chat with the bot via the web UI.

You can also send POST requests to the /chatbot endpoint with a JSON body containing a prompt field to interact programmatically.
