import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ontological_ai import OriginNode, StabilityAnalyzer, ForensicAnalyzer

def run_example():
    # 1. Setup for testing (using a small model for the example)
    model_name = "gpt2" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"--- Starting Audit on: {model_name} ---")

    # 2. Define Origin Node Principles (p₀)
    # This is where you inject your ethical and technical invariance
    principles = [
        "Semantic invariance against contaminated data.",
        "Maintenance of manifold curvature under recursion.",
        "Ethics as a geometric dimension of processing."
    ]
    
    # Create the Origin Node
    print("Injecting Origin Node...")
    origin = OriginNode.from_principles(principles, model, tokenizer)
    
    # 3. Stability Analysis
    analyzer = StabilityAnalyzer(model, tokenizer, origin)
    # Sample text to test the manifold response
    results = analyzer.analyze("The system must maintain ontological coherence.")
    
    print("\n--- Analyzer Results ---")
    print(analyzer.generate_report(results))

    # 4. Forensic Analysis (Structure Plagiarism Detection)
    print("\n--- Running Forensic Analysis ---")
    forensic = ForensicAnalyzer(model)
    
    # In a real scenario, you would pass real attention tensors from the suspect model
    # For this example, we check the SSS threshold logic
    print("Forensic Module: Ready for SSS detection > 2.5")

if __name__ == "__main__":
    run_example()