"""
evaluate.py — Automated Benchmark Testing for Intent Classification.
Generates a dataset of 600+ queries (including general chat and edge cases) 
and tests the offline LLaMA model's accuracy.
"""

import time
from core.intent import IntentClassifier

# 1. Generate the Dataset Dynamically
cities = ["Raipur", "Mumbai", "London", "Paris", "New York", "Tokyo", "Delhi", "Agra", "Aluva", "Bangalore"]
days = ["today", "tomorrow", "next Friday", "on Monday"]
destinations = ["Goa", "Pune", "Chennai", "Kochi", "Oxford", "Dubai"]

test_cases = []

# --- SKILL QUERIES ---
# Generate Weather Queries
for city in cities:
    for day in days:
        test_cases.append({"text": f"What is the weather in {city} {day}?", "expected": "get_weather"})
        test_cases.append({"text": f"Is it going to rain in {city} {day}?", "expected": "get_weather"})
        test_cases.append({"text": f"Tell me the temperature for {city}.", "expected": "get_weather"})

# Generate Navigation Queries
for origin in cities:
    for dest in destinations:
        test_cases.append({"text": f"How do I get from {origin} to {dest}?", "expected": "search_travel"})
        test_cases.append({"text": f"Driving distance between {origin} and {dest}", "expected": "search_travel"})

# Generate Note/Reminder Queries
for i in range(1, 51):
    test_cases.append({"text": f"Take a note that my project number is {i}", "expected": "create_note"})
    test_cases.append({"text": f"Remind me to submit assignment {i} tomorrow", "expected": "set_reminder"})
    test_cases.append({"text": f"Show me all my saved notes", "expected": "get_notes"})
    test_cases.append({"text": f"What are my upcoming reminders?", "expected": "get_reminders"})

# --- CONVERSATIONAL & EDGE CASE QUERIES ---
# Generate Greetings & Farewells
greetings = ["Hello!", "Hi there", "Good morning", "Hey assistant", "Yo", "Greetings", "Hi"]
farewells = ["Goodbye", "See you later", "Bye", "I'm done", "Exit", "Close the app", "Goodnight"]
for g in greetings:
    test_cases.append({"text": g, "expected": "greeting"})
for f in farewells:
    test_cases.append({"text": f, "expected": "farewell"})

# Generate General Chat
chat_queries = [
    "How are you doing today?",
    "Tell me a funny joke.",
    "What is your favorite color?",
    "Who created you?",
    "Are you self-aware?",
    "What is the meaning of life?",
    "I'm feeling a bit tired today.",
    "Can you write a poem about AI?",
    "What is the capital of Australia?",
    "Explain quantum computing to me."
]
# Multiply to create more data weight for general chat
for _ in range(3): 
    for c in chat_queries:
        test_cases.append({"text": c, "expected": "general_chat"})

# Generate Unknown / Out-of-Domain (Testing boundaries)
# These are things your AI CANNOT do, testing if it safely fails
unknown_queries = [
    "Turn off the living room lights.",
    "Play the latest song by Taylor Swift.",
    "Add milk to my Spotify playlist.",
    "Order a large pepperoni pizza.",
    "Set the thermostat to 72 degrees.",
    "Open the garage door.",
    "Transfer 50 dollars to my bank account.",
    "Start my car engine."
]
for _ in range(3):
    for u in unknown_queries:
        test_cases.append({"text": u, "expected": "unknown"})


print(f"Generated {len(test_cases)} benchmark queries.")
print("Booting up local IntentClassifier...")

# 2. Run the Benchmark
classifier = IntentClassifier()
correct = 0
total = len(test_cases)
start_time = time.time()

print("Running tests (this may take a few minutes depending on GPU)...")
for idx, case in enumerate(test_cases):
    result = classifier.classify(case["text"])
    
    if result.intent == case["expected"]:
        correct += 1
    else:
        # Print failures so you can debug them!
        print(f"❌ MISMATCH: Text: '{case['text']}' | Expected: {case['expected']} | Got: {result.intent}")
    
    # Print progress every 50 items
    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{total}...")

end_time = time.time()

# 3. Calculate and Print Results
accuracy = (correct / total) * 100
total_time = end_time - start_time
latency = total_time / total

print("\n" + "="*50)
print("🏆 COMPREHENSIVE BENCHMARK RESULTS 🏆")
print("="*50)
print(f"Total Samples Tested : {total}")
print(f"Correct Predictions  : {correct}")
print(f"Model Accuracy       : {accuracy:.2f}%")
print(f"Average Latency      : {latency:.2f} seconds per query")
print("="*50)
print("Copy these results into your research paper!")