"""
API REQUEST PARALLEL PROCESSOR (Revised for Vision Requests)

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.
It has been modified to support vision requests with model "grok-2-vision-latest".
"""

# imports
from dotenv import load_dotenv
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata

async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms sleeps so tasks can run concurrently

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if "/deployments" in request_url:
        request_header = {"api-key": f"{api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs: 0, 1, 2, ...
    status_tracker = StatusTracker()  # tracks overall progress
    next_request = None  # holds the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # when file is empty, stop reading new requests
    logging.debug("Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        requests = file.__iter__()
        logging.debug("File opened. Entering main loop.")
        async with aiohttp.ClientSession() as session:
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                    elif file_not_finished:
                        try:
                            # get new request from file
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                        except StopIteration:
                            logging.debug("Input file exhausted.")
                            file_not_finished = False

                # update available capacity based on elapsed time
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if available_request_capacity >= 1 and available_token_capacity >= next_request_tokens:
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API concurrently
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None

                # break loop if all tasks are finished and file reading is complete
                if status_tracker.num_tasks_in_progress == 0:
                    break

                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = time.time() - status_tracker.time_of_last_rate_limit_error
                if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                    remaining_seconds_to_pause = seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                    await asyncio.sleep(remaining_seconds_to_pause)
                    logging.warning(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # log final status after finishing
        logging.info(f"Parallel processing complete. Results saved to {save_filepath}")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


# dataclasses

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress."""
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and metadata."""
    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(url=request_url, headers=request_header, json=self.request_json) as response:
                response_json = await response.json()
            if "error" in response_json:
                logging.warning(f"Request {self.task_id} failed with error {response_json['error']}")
                status_tracker.num_api_errors += 1
                error = response_json
                if "rate limit" in response_json["error"].get("message", "").lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # separate count for rate limit errors
        except Exception as e:
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response_json, self.metadata]
                if self.metadata
                else [self.request_json, response_json]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions

def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment URLs
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a JSON payload to a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(request_json: dict, api_endpoint: str, token_encoding_name: str):
    """
    Count the number of tokens in the request.
    Supports completions, embeddings, and vision (grok-2-vision-latest) requests.
    """
    encoding = tiktoken.get_encoding(token_encoding_name)

    # For vision requests (grok-2-vision-latest)
    if request_json.get("model") == "grok-2-vision":
        total_tokens = 0
        # Expecting "messages" with a list of message objects
        for message in request_json.get("messages", []):
            # Each message's content is expected to be a list of items
            for item in message.get("content", []):
                if item.get("type") == "text":
                    total_tokens += len(encoding.encode(item.get("text", "")))
                elif item.get("type") == "image_url":
                    # Option: assign a fixed token cost (or ignore if desired)
                    total_tokens += 0
        return total_tokens

    # For completions requests
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # Chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows a standard format
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if name is provided, role is omitted
                        num_tokens -= 1
            num_tokens += 2  # priming for assistant reply
            return num_tokens + completion_tokens
        # Normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):
                prompt_tokens = len(encoding.encode(prompt))
                return prompt_tokens + completion_tokens
            elif isinstance(prompt, list):
                prompt_tokens = sum(len(encoding.encode(p)) for p in prompt)
                return prompt_tokens + completion_tokens * len(prompt)
            else:
                raise TypeError('Expecting a string or list of strings for "prompt" field.')

    # For embeddings requests
    elif api_endpoint == "embeddings":
        input_field = request_json["input"]
        if isinstance(input_field, str):
            return len(encoding.encode(input_field))
        elif isinstance(input_field, list):
            return sum(len(encoding.encode(i)) for i in input_field)
        else:
            raise TypeError('Expecting a string or list of strings for "input" field in embedding request.')
    
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate sequential integer task IDs: 0, 1, 2, ..."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script

if __name__ == "__main__":
    # load API key from environment variable
    load_dotenv()
    # Use XAI_API_KEY or change as needed
    api_key = os.getenv("XAI_API_KEY")

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath", required=True)
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--request_url", default="https://api.openai.com/v1/embeddings")
    parser.add_argument("--api_key", default=os.getenv("XAI_API_KEY"))
    parser.add_argument("--max_requests_per_minute", type=float, default=1500)
    parser.add_argument("--max_tokens_per_minute", type=float, default=125000)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", type=int, default=logging.INFO)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")

    # run the asynchronous processing
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            request_url=args.request_url,
            api_key=args.api_key,
            max_requests_per_minute=args.max_requests_per_minute,
            max_tokens_per_minute=args.max_tokens_per_minute,
            token_encoding_name=args.token_encoding_name,
            max_attempts=args.max_attempts,
            logging_level=args.logging_level,
        )
    )
