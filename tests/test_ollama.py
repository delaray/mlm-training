"""Test Ollama connection and list available models."""
import ollama

try:
    models = ollama.list()
    print("✅ Ollama server is running!")
    print("\nAvailable models:")
    for m in models.get('models', []):
        name = m.get('name', m.get('model', 'unknown'))
        print(f"  - {name}")

    # Test with a simple chat
    print("\n🧪 Testing chat with llama3.1:8b...")
    response = ollama.chat(
        model='llama3.1:8b',
        messages=[
            {'role': 'user', 'content': 'Say hello in exactly 5 words.'}
        ]
    )
    print(f"Response: {response['message']['content']}")
    print("\n✅ Ollama is working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure Ollama is running. Start it with: ollama serve")
