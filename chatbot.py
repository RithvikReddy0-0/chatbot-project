# Step 1 & 2: Import required tools
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Step 3: Choose model
model_name = "facebook/blenderbot-400M-distill"

# Step 4: Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 5.1: Initialize conversation history
conversation_history = []

# Step 6: Conversation loop
print("🤖 Chatbot ready! Type 'exit' or press Ctrl+C to quit.")
try:
    while True:
        # Step 5.2: Create conversation history string
        history_string = "\n".join(conversation_history)

        # Step 5.3: Get input from user
        input_text = input("\nYou: ")
        if input_text.lower() == "exit":
            print("👋 Exiting chat. Goodbye!")
            break

        # Step 5.4: Tokenize input and history
        inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

        # Step 5.5: Generate response
        outputs = model.generate(**inputs)

        # Step 5.6: Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Step 5.7: Update conversation history
        conversation_history.append(input_text)
        conversation_history.append(response)

        # Optional: Trim history to avoid excessive memory usage
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

        # Print response
        print("Bot:", response)

except KeyboardInterrupt:
    print("\n👋 Chat ended by user. Goodbye!")
