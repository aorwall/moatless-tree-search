import litellm
import os

def test_model():
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"}
        ]
        
        completion = litellm.completion(
            model="openai/Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=messages,
            temperature=0.7,
            base_url=os.getenv("CUSTOM_LLM_API_BASE"),
            api_key=os.getenv("CUSTOM_LLM_API_KEY"),
        )
        
        print("Response received:")
        print(completion.choices[0].message.content)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response}")

if __name__ == "__main__":
    test_model()