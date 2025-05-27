from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>BlenderBot Chatbot</title>
    <style>
      /* Reset and base */
      * {
        box-sizing: border-box;
      }
      body {
        background: #1a1a2e;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        color: #eee;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
      }

      /* Container */
      .chat-container {
        display: flex;
        flex-direction: column;
        width: 100%;
        max-width: 480px;
        height: 650px;
        background: #16213e;
        border-radius: 16px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.7);
        overflow: hidden;
      }

      /* Header */
      header {
        padding: 20px;
        background: #0f3460;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 1.3px;
        user-select: none;
        color: #e94560;
        text-shadow: 0 0 6px #e94560;
        position: relative;
      }

      /* Clear chat button */
      #clearBtn {
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        background: transparent;
        border: 2px solid #e94560;
        color: #e94560;
        border-radius: 20px;
        padding: 6px 14px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.3s, color 0.3s;
        user-select: none;
      }
      #clearBtn:hover {
        background: #e94560;
        color: white;
      }
      #clearBtn:active {
        transform: translateY(-50%) scale(0.95);
      }

      /* Chatbox */
      #chatbox {
        flex: 1;
        padding: 20px;
        overflow-y: auto;
        scroll-behavior: smooth;
        display: flex;
        flex-direction: column;
        gap: 14px;
        background: linear-gradient(135deg, #0f3460, #16213e);
      }

      /* Message bubbles */
      .message {
        max-width: 75%;
        padding: 12px 18px;
        border-radius: 24px;
        line-height: 1.4;
        font-size: 1rem;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        animation: fadeIn 0.3s ease forwards;
        position: relative;
      }
      .user {
        align-self: flex-end;
        background: #e94560;
        color: white;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.5);
      }
      .bot {
        align-self: flex-start;
        background: #0f3460;
        color: #a0c4ff;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 12px rgba(15, 52, 96, 0.6);
        font-style: italic;
      }

      /* Timestamps */
      .timestamp {
        position: absolute;
        bottom: -18px;
        font-size: 0.7rem;
        color: #aaa;
        opacity: 0.7;
        user-select: none;
      }
      .user .timestamp {
        right: 12px;
      }
      .bot .timestamp {
        left: 12px;
      }

      /* Typing indicator */
      #typingIndicator {
        font-style: italic;
        color: #aaa;
        padding: 0 20px 12px 20px;
        height: 22px;
        user-select: none;
      }

      /* Input area */
      .input-area {
        display: flex;
        padding: 15px 20px;
        background: #0f3460;
        border-top: 1px solid #e94560;
      }

      input#prompt {
        flex: 1;
        border: none;
        padding: 12px 18px;
        border-radius: 30px;
        font-size: 1.1rem;
        outline: none;
        transition: box-shadow 0.3s ease;
        color: #fff;
        background: #1a1a2e;
      }
      input#prompt::placeholder {
        color: #888;
        font-style: italic;
      }
      input#prompt:focus {
        box-shadow: 0 0 10px #e94560;
      }

      button {
        margin-left: 15px;
        padding: 0 25px;
        background: #e94560;
        border: none;
        border-radius: 30px;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: background 0.3s ease;
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.6);
      }
      button:hover {
        background: #ff3366;
        box-shadow: 0 6px 16px rgba(255, 51, 102, 0.7);
      }
      button:active {
        transform: scale(0.97);
      }

      /* Scrollbar style */
      #chatbox::-webkit-scrollbar {
        width: 8px;
      }
      #chatbox::-webkit-scrollbar-track {
        background: #16213e;
      }
      #chatbox::-webkit-scrollbar-thumb {
        background: #e94560;
        border-radius: 4px;
      }

      /* Fade in animation for messages */
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px);}
        to { opacity: 1; transform: translateY(0);}
      }

      /* Responsive */
      @media (max-width: 520px) {
        .chat-container {
          height: 100vh;
          border-radius: 0;
          max-width: 100%;
        }
      }
    </style>
  </head>
  <body>
    <div class="chat-container">
      <header>
        BlenderBot Chatbot
        <button id="clearBtn" title="Clear Chat">Clear</button>
      </header>
      <div id="chatbox"></div>
      <div id="typingIndicator"></div>
      <div class="input-area">
        <input id="prompt" placeholder="Say something..." autocomplete="off" autofocus />
        <button onclick="sendMessage()">Send</button>
      </div>
    </div>

    <script>
      const chatbox = document.getElementById("chatbox");
      const promptInput = document.getElementById("prompt");
      const typingIndicator = document.getElementById("typingIndicator");
      const clearBtn = document.getElementById("clearBtn");

      // Format time as HH:MM am/pm
      function getCurrentTime() {
        const d = new Date();
        let hours = d.getHours();
        const minutes = d.getMinutes().toString().padStart(2, '0');
        const ampm = hours >= 12 ? 'pm' : 'am';
        hours = hours % 12 || 12;
        return hours + ':' + minutes + ' ' + ampm;
      }

      function appendMessage(sender, text, className) {
        const time = getCurrentTime();
        const msgDiv = document.createElement("div");
        msgDiv.classList.add("message", className);
        msgDiv.innerHTML = `<strong>${sender}:</strong> ${text} <span class="timestamp">${time}</span>`;
        chatbox.appendChild(msgDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
      }

      function showTyping(show) {
        typingIndicator.textContent = show ? "Bot is typing..." : "";
      }

      async function sendMessage() {
        const input = promptInput.value.trim();
        if (!input) return;

        appendMessage("You", input, "user");
        promptInput.value = "";
        promptInput.focus();

        showTyping(true);

        try {
          const response = await fetch("/chatbot", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: input })
          });

          showTyping(false);

          if (!response.ok) throw new Error("Server error");

          const data = await response.json();
          appendMessage("Bot", data.response, "bot");
        } catch (error) {
          showTyping(false);
          appendMessage("Bot", "Error: Could not reach server.", "bot");
        }
      }

      // Send on Enter key
      promptInput.addEventListener("keydown", function(event) {
        if (event.key === "Enter") {
          event.preventDefault();
          sendMessage();
        }
      });

      // Clear chat on button click
      clearBtn.addEventListener("click", () => {
        chatbox.innerHTML = "";
        promptInput.focus();
      });
    </script>
  </body>
</html>
"""



@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    try:
        data = request.get_json()
        input_text = data.get('prompt', '').strip()

        if not input_text:
            return jsonify({'response': 'Please enter something.'})

        # Combine conversation history into one string
        history = "\n".join(conversation_history)
        inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=60)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Update conversation history
        conversation_history.append(input_text)
        conversation_history.append(response)

        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False)
