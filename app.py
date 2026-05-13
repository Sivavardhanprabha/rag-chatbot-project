import os
from groq import Groq
from dotenv import load_dotenv

# 1. This line is the 'Bridge'. 
# It tells Python to go look at your .env file and load the variables inside.
load_dotenv()

# 2. This line grabs the specific value of GROQ_API_KEY from the vault.
api_key = os.getenv("GROQ_API_KEY")

# 3. This initializes the Groq 'Client'. 
# This is like opening a phone line to the AI server.
client = Groq(api_key=api_key)

# 4. We are now sending a 'Request' to the AI.
try:
    print("Testing connection to Groq...")
    
    # We ask the Llama-3 model to reply to us
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Confirm 'System Online' if you can hear me.",
            }
        ],
        model="llama-3.1-8b-instant",
    )

    # 5. This prints the AI's answer to your terminal
    print("\nAI says: " + chat_completion.choices[0].message.content)
    print("\n--- Success! The Gateway is open. ---")

except Exception as e:
    # If something is wrong (like a bad API key), this will tell us
    print(f"\nSomething went wrong: {e}")