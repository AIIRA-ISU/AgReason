import os
import re
import ast
import json
import time
import asyncio
import aiohttp
import pandas as pd
from tqdm import tqdm
from collections import deque
from filteration_prompt import prompt

# Load your keys
keys = json.load(open("keys.json", "r"))
API_KEY = keys["together"]

TOGETHER_ENDPOINT = "https://api.together.xyz/v1/chat/completions"

MAX_CONCURRENT_REQUESTS = 30
REQUEST_TIMESTAMPS = deque()

async def enforce_rate_limit():
    while True:
        now = time.time()
        while REQUEST_TIMESTAMPS and now - REQUEST_TIMESTAMPS[0] > 1:
            REQUEST_TIMESTAMPS.popleft()
        if len(REQUEST_TIMESTAMPS) < MAX_CONCURRENT_REQUESTS:
            REQUEST_TIMESTAMPS.append(now)
            return
        await asyncio.sleep(0.05)


def create_prompt(question, task):
    system_prompts = {
        "paraphrase": prompt.SYSTEM_PROMPT_PARAPHRASE,
        "filter": prompt.SYSTEM_PROMPT_FILTER
    }
    user_prompts = {
        "paraphrase": prompt.USER_PROMPT_PARAPHRASE,
        "filter": prompt.USER_PROMPT_FILTER
    }

    sys_p = system_prompts[task].format(question)
    usr_p = user_prompts[task].format(question)
    return sys_p, usr_p

def process_response(response_json, task):
    content = response_json.get("response", "")

    if task == "strip_think":
        match = re.search(r"</think>\s*(.*)", content, re.DOTALL)
        text_after_think = match.group(1).strip() if match else content.strip()
        return text_after_think

    elif task == "jsonify":
        try:
            return ast.literal_eval(content)
        except:
            return False
    else:
        return content.strip()

async def chat_completion(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    question: str,
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    top_p: float = 0.95,
    stop=None,
):
    """
    Make a single POST call to  chat completion endpoint, subject to
    both concurrency and rate-limiting.
    """
    if model == "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": 
        system_prompt = None
        user_prompt = [{"type": "text", "text": user_prompt}]
        
    async with sem:
        await enforce_rate_limit()

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        if system_prompt is not None:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "top_p": top_p
            }
        else:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "top_p": top_p
            }

        try:
            async with session.post(TOGETHER_ENDPOINT, headers=headers, json=payload) as resp:
                retry_after = resp.headers.get("Retry-After")
                if resp.status == 429 and retry_after:
                    print(f"Rate limited on question: '{question[:50]}...' | Retrying after {retry_after} seconds")
                    await asyncio.sleep(float(retry_after))
                    return await chat_completion(
                    session,
                    sem,
                    system_prompt,
                    user_prompt,
                    question,
                    model,
                    temperature,
                    max_tokens,
                    top_p
                )

                resp_json = await resp.json()
                if resp.status != 200:
                    return {"question": question, "error": resp_json.get("error", "Unknown error")}
                return {
                    "question": question,
                    "response": resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                }
        except Exception as e:
            return {"question": question, "error": str(e)}

async def filter_single_question(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    question: str,
    initial_model_id: str,
    second_model_id: str,
):
    # 1. Paraphrase
    sys_p, usr_p = create_prompt(question, task="paraphrase")
    response_json = await chat_completion(
        session=session,
        sem=sem,
        system_prompt=sys_p,
        user_prompt=usr_p,
        question=question,
        model=initial_model_id,
        temperature=1,
        max_tokens=4096,
        top_p=0.95
    )  # need to check the specifications

    if "error" in response_json:
        return {
            "filtered_out": True,
            "original_question": question,
            "processed_question": None,
            "error": response_json["error"]
        }

    processed_response = process_response(response_json, task="jsonify")
    if not processed_response:
        return {
            "filtered_out": True,
            "original_question": question,
            "processed_question": None,
            "error": "Failed to process in the initial layer"
        }
    
    paraphrased_question = processed_response.get("paraphrased_text", None) # this is the key that needs to be mentioned in the prompt.

    if paraphrased_question is None:
        return {
            "filtered_out": True,
            "original_question": question,
            "processed_question": None,
            "error": "No key `paraphrased_text` in the response"
        }
    
    # 2. Filter Question
    sys_p, usr_p = create_prompt(paraphrased_question, task="filter")
    response_json = await chat_completion(   
        session=session,
        sem=sem,
        system_prompt=sys_p,
        user_prompt=usr_p,
        question=question,
        model=second_model_id,
        temperature=1,
        max_tokens=4096,
        top_p=0.95
    ) # need to check the specifications

    if "error" in response_json:
        return {
            "filtered_out": True,
            "original_question": question,
            "processed_question": None,
            "error": response_json["error"]
        }

    processed_response = process_response(response_json, task="jsonify")

    if not processed_response:
        return {
            "filtered_out": True,
            "original_question": question,
            "processed_question": None,
            "error": "Failed to process in the second layer"
        }
    
    decision = processed_response.get("decision", None)   # this is the key that needs to be mentioned in the prompt.
    if decision is None:
        return {
            "filtered_out": True,
            "original_question": question,
            "processed_question": None,
            "error": "No key `decision` in the response"
        }
    
    if decision=="True":
        return {
            "filtered_out": False,
            "original_question": question,
            "processed_question": paraphrased_question
        }

    return {
        "filtered_out": True,
        "original_question": question,
        "processed_question": processed_response,
        "error": "Failed in the final layer. Incorrect Question."
    }


def load_questions_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df['questions'].dropna().tolist()

async def main():
    data_path = "new_generations/flowchart/80k.xlsx"
    partial_file_path = "new_filtered_generation/flowchart/partial_results.jsonl"

    os.makedirs("new_filtered_generation", exist_ok=True)

    # Qwen-2.5 is used for paraphrasing the unstructured question coming from the flowchart.
    # LLama-4-Maverick is used for filtering the question based on various Agricultural factual checks.
    initial_model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
    second_model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    

    initial_filtered_out = []
    filtered_question = []
    results = []

    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        questions = load_questions_from_excel(data_path)
        for q in questions:
            tasks.append(
                filter_single_question(
                    sem,
                    session,
                    question=q,
                    initial_model_id=initial_model_id,
                    second_model_id=second_model_id
                )
            )

        print("Processing questions in parallel...")
        pbar = tqdm(total=len(tasks), desc="Processing")

        with open(partial_file_path, "a", encoding="utf-8") as partial_file:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                partial_file.write(json.dumps(result, ensure_ascii=False) + "\n")

                pbar.update(1)
        pbar.close()
        print("All tasks finished.")

    for r in results:
        if r["filtered_out"]:
            initial_filtered_out.append({
                "original_question": r["original_question"],
                "processed_question": r["processed_question"],
                "error": r.get("error", None)
            })
        else:
            filtered_question.append({
                "question": r["processed_question"]
            })


    df = pd.DataFrame(initial_filtered_out)
    df.to_excel("new_filtered_generation/flowchart/filtered_out.xlsx", index=False)


    df = pd.DataFrame(filtered_question)
    df.to_excel("new_filtered_generation/flowchart/kept_questions.xlsx", index=False)


    print("Done!")
    print(f"  - Filtered out: {len(initial_filtered_out)}")
    print(f"  - Kept: {len(filtered_question)}")

if __name__ == "__main__":
    asyncio.run(main())

