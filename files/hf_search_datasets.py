from huggingface_hub import HfApi

keywords = ["jailbreak","jailbreaks","jailbreak_prompt","jailbreak-prompts",
            "prompt-injection","prompt injection","jailbreak_prompts","injection","jailbreak"]

print("Searching Hugging Face datasets for keywords:", keywords)
api = HfApi()
found = set()
for k in keywords:
    try:
        results = api.list_datasets(search=k, sort="lastModified")
    except Exception as e:
        print(f"  Error searching for '{k}': {e}")
        continue
    for r in results:
        found.add(r.id)

found = sorted(found)
print(f"Found {len(found)} candidate datasets:\n")
for d in found:
    print(d)
