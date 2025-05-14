import argparse
import asyncio
import aiohttp
import pandas as pd
import random
import time
import json
from collections import deque
from tqdm import tqdm
from google import genai
from google.genai import types



MAX_CONCURRENT_REQUESTS = 3
REQUEST_TIMESTAMPS = deque()

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_API_KEY = json.load(open("keys.json"))['together']

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_KEY = json.load(open("keys.json"))['claude']

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = json.load(open("keys.json"))['gpt']

GEMINI_API_KEY = json.load(open("keys.json"))['gemini']


async def enforce_rate_limit():
    while True:
        now = time.time()
        while REQUEST_TIMESTAMPS and now - REQUEST_TIMESTAMPS[0] > 1:
            REQUEST_TIMESTAMPS.popleft()
        if len(REQUEST_TIMESTAMPS) < MAX_CONCURRENT_REQUESTS:
            REQUEST_TIMESTAMPS.append(now)
            return
        await asyncio.sleep(0.05)

async def fetch_completion(session, sem, question, system_prompt, model_name, provider="together", temperature=0.6, top_p=0.95, max_token=12000, not_thinking=False):
    async with sem:
        await enforce_rate_limit()

        if provider == "together":
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }
            if system_prompt is not None:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
            else:
                messages = [{"role": "user", "content": question}]
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_completion_tokens": max_token
            }
            url = TOGETHER_API_URL

        elif provider == "claude":
            headers = {
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            messages = [{"role": "user", "content": question}]
            
            if not not_thinking:
                payload = {
                    "model": model_name,
                    "max_tokens": 4096,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 3096
                    },
                    "messages": messages
                }
            else:
                payload = {
                    "model": model_name,
                    "max_tokens": 4096,
                    "messages": messages
                }

            if system_prompt:
                payload["messages"][0]["content"] = system_prompt + "\n\n" + question

            url = CLAUDE_API_URL
            
        elif provider == "gpt":
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            if system_prompt is not None:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
            else:
                messages = [{"role": "user", "content": question}]
            payload = {
                "model": model_name,
                "messages": messages,
            }
            url = OPENAI_API_URL

        elif provider == "gemini":
            def call_gemini():
                client = genai.Client(api_key=GEMINI_API_KEY)
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=system_prompt + "\n\n" + question if system_prompt else question)],
                    ),
                ]
                generate_content_config = types.GenerateContentConfig(
                    response_mime_type="text/plain",
                )
                chunks = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=generate_content_config,
                )
                return ''.join(chunk.text or "" for chunk in chunks)

            try:
                response_text = await asyncio.to_thread(call_gemini)
                return {"question": question, "response": response_text}
            except Exception as e:
                return {"question": question, "error": str(e)}


        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                retry_after = resp.headers.get("Retry-After")
                if resp.status == 429 and retry_after:
                    print(f"Rate limited on question: '{question[:50]}...' | Retrying after {retry_after} seconds")
                    await asyncio.sleep(float(retry_after))
                    return await fetch_completion(session, sem, question, system_prompt, model_name, provider, temperature, top_p, not_thinking)

                resp_json = await resp.json()
                if resp.status != 200:
                    return {"question": question, "error": resp_json.get("error", "Unknown error")}

                if provider == "claude":
                    if not not_thinking:
                        response = "<think>{}</think>{}".format(resp_json.get("content", "")[0].get("thinking", ""), resp_json.get("content", "")[1].get("text", ""))
                    else:
                        response = resp_json.get("content", "")[0].get("text", "")
                    return {
                        "question": question,
                        "response": response
                    }
                
                elif provider == "gpt":
                    return {
                        "question": question,
                        "response": resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    }
                
                else:
                    return {
                        "question": question,
                        "response": resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    }
                
        except Exception as e:
            return {"question": question, "error": str(e)}


def load_from_benchmark_file(file_path, num_sample):
    data = json.load(open(file_path))
    questions = [d["processed_question"] for d in data]

    if num_sample is not None:
        return random.sample(questions, num_sample)
    else:
        return questions

async def main(args):
    if not args.from_benchmark:
        df = pd.read_excel(args.input_file)

        if args.num_samples is not None:
            sampled_questions = random.sample(df["question"].dropna().tolist(), args.num_samples)
        else:
            sampled_questions = df["question"].dropna().tolist()
    else:
        sampled_questions = load_from_benchmark_file(args.input_file, args.num_samples)

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    results = []

    system_prompt = json.load(open("system_prompt.json")).get(args.model, None)

    async with aiohttp.ClientSession() as session:
        for question in sampled_questions:
            tasks.append(fetch_completion(
                session, sem, question, system_prompt, args.model,
                provider=args.provider, temperature=0.6, top_p=0.95, not_thinking=args.not_thinking
            ))
        
        pbar = tqdm(total=len(tasks), desc="Processing")
        with open(args.partial_file_path, "a", encoding="utf-8") as partial_file:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                partial_file.write(json.dumps(result, ensure_ascii=False) + "\n")

                pbar.update(1)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done! Responses saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async API Caller for Together API")
    parser.add_argument("--input_file", type=str, required=True, help="Path to Excel file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results")
    parser.add_argument("--partial_file_path", type=str, default="partial_results.jsonl", help="Path to save partial results")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1", help="Model name to use")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of questions to sample")
    parser.add_argument("--from_benchmark", action="store_true", help="Whether to use the benchmark dataset")
    parser.add_argument("--provider", type=str, default="together", choices=["together", "claude", "gpt", "gemini"], help="API provider")
    parser.add_argument("--not_thinking", action="store_true", help="Whether to remove the thinking part in the response")



    args = parser.parse_args()
    asyncio.run(main(args))
