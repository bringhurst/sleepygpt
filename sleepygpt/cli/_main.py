def chat_and_learn(model, device):
    print("SleepyGPT Chatbot ready! Start chatting. Type 'quit' to exit.")
    previous_input = None
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        if previous_input is not None:
            # Update the model based on the conversation
            update_from_chat(model, optimizer, device, previous_input, user_input)
        
        previous_input = user_input  # Save the last input for learning
