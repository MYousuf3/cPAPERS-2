import os
import json
import re
import time
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def openai_inference(model: str, system_prompt: str, prompt: str):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" }
    )
    return response.choices[0].message.content

def get_review_split_prompt_llama(review: str):
    """
    Creates a prompt for splitting a review into individual review points/pieces.
    """
    prompt = 'Given the example Review below, look at how it is split into individual review points or pieces. Each point represents a distinct comment, question, strength, weakness, or observation. Now, split the provided Review into similar individual points.'
    
    example_review = """
    Originality: 
    The algorithm seems to be novel and differs from previous comparable approaches like CoreSet or BADGE. Related work is adequately cited. 
    
    Quality: 
    The authors show empirically that their algorithm, Cluster-Margin, is both more efficient (O(nlog(n) vs O(n√(n)) than CoreSet and BADGE in practice and more effective. In particular, the algorithm clearly outperforms CoreSet, BADGE, Margin and Random on the Open Images dataset. The algorithm requires 29% less labels than the second-best model in the 100k batch-size setting and 60% less labels in the 1M batch-size setting to achieve the same result (mAP). Cluster-Margin also outperforms all other methods on CIFAR10, CIFAR100 and obtains a similar performance on SVHN. 
    
    The authors also establish a theoretical guarantee for the Cluster-MarginV algorithm and show that those results hold for the Cluster-Margin algorithm in specific settings. In particular, they show that the Cluster-MarginV algorithm has a label complexity bound which improves over the Margin algorithm by a factor beta. They also show that this improvement is possible, under specific hypotheses like an optimal volume-based sampler, when the dimensionality of the embedding space is small or when the batch size k is large. They also show that the optimal volume-based sampler is approximately equivalent to the Cluster-Margin algorithm. They finally show that log(k) is an upper bound on the improvement of query complexity for any sampler. 
    The authors are aware and mention that their theoretical results are initial and that equating volume based samplers and the Cluster-Margin algorithm is an open research question. 
    
    Clarity: 
    The paper is very clear and well organized. The authors detail the hyper-parameters and compute details used for the experiments. The Cluster-Margin algorithm is also explained in detail.  
    
    Significance: 
    The results are important as the algorithm allows for more efficient and effective large-batch-size active learning compared to existing methods. The authors also provide initial theoretical guarantees to explain the improvements obtained with the Cluster-Margin algorithm. 
    """
    
    example_points = [
        "The algorithm seems to be novel and differs from previous comparable approaches like CoreSet or BADGE. Related work is adequately cited.",
        "The authors show empirically that their algorithm, Cluster-Margin, is both more efficient (O(nlog(n) vs O(n√(n)) than CoreSet and BADGE in practice and more effective.",
        "The algorithm clearly outperforms CoreSet, BADGE, Margin and Random on the Open Images dataset. The algorithm requires 29% less labels than the second-best model in the 100k batch-size setting and 60% less labels in the 1M batch-size setting to achieve the same result (mAP).",
        "Cluster-Margin also outperforms all other methods on CIFAR10, CIFAR100 and obtains a similar performance on SVHN.",
        "The authors establish a theoretical guarantee for the Cluster-MarginV algorithm and show that those results hold for the Cluster-Margin algorithm in specific settings.",
        "The Cluster-MarginV algorithm has a label complexity bound which improves over the Margin algorithm by a factor beta.",
        "This improvement is possible, under specific hypotheses like an optimal volume-based sampler, when the dimensionality of the embedding space is small or when the batch size k is large.",
        "The optimal volume-based sampler is approximately equivalent to the Cluster-Margin algorithm.",
        "log(k) is an upper bound on the improvement of query complexity for any sampler.",
        "The authors are aware and mention that their theoretical results are initial and that equating volume based samplers and the Cluster-Margin algorithm is an open research question.",
        "The paper is very clear and well organized. The authors detail the hyper-parameters and compute details used for the experiments. The Cluster-Margin algorithm is also explained in detail.",
        "The results are important as the algorithm allows for more efficient and effective large-batch-size active learning compared to existing methods. The authors also provide initial theoretical guarantees to explain the improvements obtained with the Cluster-Margin algorithm."
    ]
    
    content = (
        f'[Context]\n{prompt}\n\n'
        f'[Example Review]\n{example_review}\n\n'
        f'[Example Split Points]\n{json.dumps({"review_points": example_points}, indent=2)}\n\n'
        f'[Review to Split]\n{review}\n')
    
    system_prompt = "You are a helpful assistant that splits academic paper reviews into individual review points. Each point should be a distinct comment, observation, strength, weakness, or question. Your response should be in JSON format with a 'review_points' key containing an array of strings. Do not add any other unnecessary content in your response."

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    formatted_input = f"{B_SYS}{system_prompt}{E_SYS}{B_INST}{content}{E_INST}"
    return formatted_input

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def open_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def split_reviews_vllm(or_data, path, tokenizer, llm, output_file="split_reviews_raw.json"):
    """
    Split reviews using VLLM inference
    """
    sampling_params = SamplingParams(
            top_k=10,
            n=1,
            max_tokens=2000,
            stop_token_ids=[tokenizer.eos_token_id])

    prompt_batch = [get_review_split_prompt_llama(item["review"]) for item in or_data]

    try:
        outputs = llm.generate(prompt_batch, sampling_params=sampling_params)
    except Exception as e:
        print(f"Error during llm.generate: {e}")
        outputs = []
    
    for idx, output in enumerate(outputs):
        if hasattr(output, 'outputs') and output.outputs:
            text = output.outputs[0].text
            or_data[idx]["split_review_raw"] = text
        else:
            print(f"No outputs for prompt {idx}")
            or_data[idx]["split_review_raw"] = None

    save_json(or_data, f"{path}/{output_file}")
    return or_data

def split_reviews_openai(data, path, output_file="split_reviews_openai.json"):
    """
    Split reviews using OpenAI API (GPT)
    """
    for idx, item in enumerate(tqdm(data)):
        if "review" not in item or not item["review"]:
            data[idx]["split_review_points"] = None
            continue
            
        try:
            # Create a simpler prompt for OpenAI
            prompt = f"""Split the following academic paper review into individual review points. Each point should be a distinct comment, observation, strength, weakness, or question. Preserve the original text as much as possible.

Review:
{item["review"]}

Return your response as a JSON object with a 'review_points' key containing an array of strings."""
            
            system_prompt = """You are a helpful assistant that splits academic paper reviews into individual review points. Return your response in JSON format with a 'review_points' key containing an array of strings."""
            
            response = openai_inference("gpt-3.5-turbo", system_prompt, prompt)
            data[idx]["split_review_points"] = response
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            data[idx]["split_review_points"] = None

        # Save periodically to avoid data loss
        if (idx + 1) % 100 == 0:
            save_json(data, f"{path}/{output_file}")
        
        time.sleep(0.5)  # Rate limiting

    save_json(data, f"{path}/{output_file}")
    return data

def clean_with_gpt(data, path, output_file="gpt_cleaned_reviews.json"):
    """
    Use GPT-3.5-turbo to clean messy LLM output and extract review points
    """
    count_cleaned = 0
    
    for idx, item in enumerate(tqdm(data)):
        raw_output = item.get("split_review_raw") or item.get("split_review_points")
        
        if raw_output:
            try:
                # Use GPT to extract clean JSON from messy output
                system_prompt = """You are a helpful assistant that extracts clean JSON data from messy text. 
Extract the review points from the provided text and return them as a clean JSON object with a 'review_points' key containing an array of strings.
Only include the actual review points, ignore any formatting tokens, prompt text, or other noise."""
                
                prompt = f"""Extract the review points from this messy output and return clean JSON:

{raw_output}

Return ONLY a JSON object in this format:
{{"review_points": ["point 1", "point 2", ...]}}"""
                
                response = openai_inference("gpt-3.5-turbo", system_prompt, prompt)
                
                # Try to parse the GPT response
                parsed = json.loads(response)
                if "review_points" in parsed and isinstance(parsed["review_points"], list):
                    data[idx]["cleaned_review_points"] = parsed["review_points"]
                    count_cleaned += 1
                else:
                    print(f"Item {idx}: GPT response missing review_points")
                    data[idx]["cleaned_review_points"] = None
                    
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                data[idx]["cleaned_review_points"] = None
        else:
            data[idx]["cleaned_review_points"] = None
        
        # Save periodically to avoid data loss
        if (idx + 1) % 50 == 0:
            save_json(data, f"{path}/{output_file}")
            print(f"Progress: {idx + 1}/{len(data)} - Successfully cleaned: {count_cleaned}")
        
        # Rate limiting
        time.sleep(0.5)
    
    print(f"Successfully cleaned {count_cleaned} out of {len(data)} reviews")
    save_json(data, f"{path}/{output_file}")
    return data

def clean_split_reviews(data, path, output_file="cleaned_split_reviews.json"):
    """
    Parse and clean the split review points (basic JSON parsing)
    """
    cleaned_data = []
    count_cleaned = 0
    
    for idx, item in enumerate(tqdm(data)):
        # Check for both VLLM output (split_review_raw) and OpenAI output (split_review_points)
        raw_output = item.get("split_review_raw") or item.get("split_review_points")
        
        if raw_output:
            try:
                parsed = json.loads(raw_output)
                if "review_points" in parsed and isinstance(parsed["review_points"], list):
                    data[idx]["cleaned_review_points"] = parsed["review_points"]
                    count_cleaned += 1
                else:
                    data[idx]["cleaned_review_points"] = None
            except Exception as e:
                print(f"Error parsing item {idx}: {e}")
                data[idx]["cleaned_review_points"] = None
        else:
            data[idx]["cleaned_review_points"] = None
    
    print(f"Successfully cleaned {count_cleaned} out of {len(data)} reviews")
    save_json(data, f"{path}/{output_file}")
    return data

def main():
    base_path = './neurips/2021/'
    
    # Load raw QA data
    print("Loading raw_qas.json...")
    or_data = open_json(base_path + "raw_qas.json")
    print(f"Loaded {len(or_data)} entries")
    
    # Take only the top 1000 entries
    or_data = or_data[:1000]
    print(f"Processing top {len(or_data)} entries")
    
    # Fix CUDA multiprocessing issues
    # The error "CUDA driver initialization failed" happens because worker processes
    # can't access CUDA when using tensor_parallel_size > 1
    os.environ['VLLM_USE_V1'] = '0'  # Use legacy engine (v0) instead of v1
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    # Option 1: Use VLLM
    model = "meta-llama/Meta-Llama-3-8B-Instruct"
    gpus = 4  
    
    print(f"Loading tokenizer for {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
    print(f"Initializing LLM with {gpus} GPU(s)...")
    print("Downloading and loading model (this may take a few minutes)...")
    llm = LLM(
            model, 
            tokenizer_mode="auto",
            tensor_parallel_size=gpus,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.95)
    
    print("Splitting reviews using VLLM...")
    output_file = "split_reviews_top1000.json"
    split_data = split_reviews_vllm(or_data, base_path, tokenizer, llm, output_file)

    
    # Clean and parse the results using GPT-3.5-turbo
    print("Cleaning split reviews with GPT-3.5-turbo...")
    cleaned_output_file = "gpt_cleaned_reviews_top1000.json"
    cleaned_data = clean_with_gpt(split_data, base_path, cleaned_output_file)
    
    # Alternative: Use basic JSON parsing (uncomment to use instead of GPT cleaning)
    """
    print("Cleaning split reviews with basic parsing...")
    cleaned_output_file = "cleaned_split_reviews_top1000.json"
    cleaned_data = clean_split_reviews(split_data, base_path, cleaned_output_file)
    """
    
    print(f"Done! Processed {len(cleaned_data)} reviews")
    print(f"Output saved to: {base_path}{cleaned_output_file}")

def clean_existing_file():
    """
    Standalone function to clean an already existing split_reviews file using GPT-3.5-turbo
    Useful if you already have the VLLM output and just want to clean it
    """
    base_path = './neurips/2021/'
    input_file = "split_reviews_top1000.json"
    output_file = "gpt_cleaned_reviews_top1000.json"
    
    print(f"Loading {input_file}...")
    data = open_json(base_path + input_file)
    print(f"Loaded {len(data)} entries")
    
    print("Cleaning with GPT-3.5-turbo...")
    cleaned_data = clean_with_gpt(data, base_path, output_file)
    
    print(f"Done! Cleaned {len(cleaned_data)} reviews")
    print(f"Output saved to: {base_path}{output_file}")

if __name__ == "__main__":
    # Run the full pipeline (VLLM splitting + GPT cleaning)
    # main()
    
    # Or, to only clean an existing file, uncomment this and comment out main():
    clean_existing_file()

