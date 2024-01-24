def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Process the input, predict the response using the model, and print the response
        # ...

chat()
