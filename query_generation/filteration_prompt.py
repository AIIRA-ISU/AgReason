class prompt:
    SYSTEM_PROMPT_PARAPHRASE = """"""
    SYSTEM_PROMPT_FILTER = """"""

    USER_PROMPT_PARAPHRASE = """
You will be provided with a short, unstructured text that contains fragmented or loosely phrased information related to agriculture. This may include details about crops, soil health, weather conditions, pests, diseases, geographic location, or farm practices.
Your task is to convert the input into a clear, well-formed question that reflects the original intent. If the original text is not phrased as a question, infer the implied question based on the context.

Important: Preserve all relevant details (e.g., crop type, location, soil condition, farming method). Do not omit or alter factual content.
Output format:

{{
  "paraphrased_text": "<your rephrased question here>"
}}

Here is the input: "{}"
"""

    USER_PROMPT_FILTER = """
You will be provided with a question that includes agronomic information such as crop type, location, soil health, weather, pests, diseases, or farming practices.
Your task is to evaluate whether the information in the question is:
* Factually correct
* Internally coherent
* Practically feasible in context
* Consistent with known agronomic relationships
Specifically, assess whether the following associations are valid:
* Crop and geographic region (is the crop typically grown there?)
* Scale of cultivation (is the scale appropriate for the crop in that region?)
* Weather conditions (are they plausible for the stated region and season?)
* Pests or diseases (are they common for that crop and region?)
* Soil conditions and farming practices (do they align with agronomic best practices for that context?)

Important: If any major element is implausible, inconsistent, or inaccurate, return "decision": "False". Otherwise, return "decision": "True".

Output format:

{{
  "decision": "True"  // or "False",
  "reason": "Brief explanation of the key agronomic justification for the decision."
}}

Strict Requirement: Only return the JSON object above. Do not include any explanation, commentary, or extra text.

Here is the question to evaluate: “{}”
"""