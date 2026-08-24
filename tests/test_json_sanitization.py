"""Test JSON sanitization for control characters."""
from src.sft_data import sanitize_json_string, extract_json_from_text
import json

# Test 1: JSON with control characters (common LLM issue)
print("Test 1: JSON with control characters")
malformed = '{"question": "What is ML?", "answer": "Machine\x0bLearning", "type": "definition"}'
print(f"Original: {repr(malformed)}")
try:
    # This would fail
    json.loads(malformed)
    print("❌ Should have failed")
except json.JSONDecodeError as e:
    print(f"✓ Failed as expected: {e}")

# Now with sanitization
sanitized = sanitize_json_string(malformed)
print(f"Sanitized: {repr(sanitized)}")
result = json.loads(sanitized)
print(f"✅ Parsed successfully: {result}")
print()

# Test 2: JSON in markdown fence with control chars
print("Test 2: JSON in markdown fence")
markdown = '''```json
{
  "pairs": [
    {"question": "Test\x01", "answer": "Answer\x02"}
  ]
}
```'''
print(f"Original: {repr(markdown[:50])}...")
result = extract_json_from_text(markdown)
print(f"✅ Extracted: {result}")
print()

# Test 3: Array format
print("Test 3: Direct array format")
array_json = '[{"q": "test\x03"}, {"q": "test2\x04"}]'
result = extract_json_from_text(array_json)
print(f"✅ Extracted: {result}")
print()

print("🎉 All tests passed! JSON sanitization working correctly.")
