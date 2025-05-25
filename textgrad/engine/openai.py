try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")

import os
import json
import base64
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes

# Default base URL for OLLAMA
OLLAMA_BASE_URL = 'http://localhost:11434/v1'
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

# Check if the user set the OLLAMA_BASE_URL environment variable
if os.getenv("OLLAMA_BASE_URL"):
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

class ChatOpenAI(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str="gpt-3.5-turbo-0613",
        system_prompt: str=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        base_url: str=None,
        site_url: str=None,  # Added for OpenRouter
        site_name: str=None,  # Added for OpenRouter
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param base_url: Used to support Ollama and OpenRouter
        :param site_url: Used for OpenRouter's HTTP-Referer header
        :param site_name: Used for OpenRouter's X-Title header
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")

        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.base_url = base_url
        
        # Default OpenRouter URL
        OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
        
        if not base_url:
            # Standard OpenAI setup
            if os.getenv("OPENAI_API_KEY") is None:
                raise ValueError("Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")
            
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif base_url == OLLAMA_BASE_URL:
            # Ollama setup
            self.client = OpenAI(
                base_url=base_url,
                api_key="ollama"
            )
        elif base_url == OPENROUTER_BASE_URL:
            # OpenRouter setup
            api_key = os.getenv("OPEN_ROUTER_API_KEY")
            if not api_key:
                raise ValueError("Please set the OPEN_ROUTER_API_KEY environment variable to use OpenRouter.")
                
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": site_url if site_url else "",
                    "X-Title": site_name if site_name else "",
                }
            )
        else:
            raise ValueError("Invalid base URL provided. Please use one of the supported base URLs: OLLAMA_BASE_URL or OpenRouter URL.")

        self.model_string = model_string
        self.is_multimodal = is_multimodal

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(
        self,
        content: Union[str, bytes, List[Union[str, bytes]]],
        system_prompt: str | None = None,
        **kwargs
    ):
        # ---------------- text OR single image  ----------------
        if isinstance(content, (str, bytes)):          # <─  add bytes here
            # Guard-rail: image bytes need a vision model
            if isinstance(content, bytes) and not self.is_multimodal:
                raise NotImplementedError(
                    "Multimodal generation is not enabled for this engine."
                )
            return self._generate_from_single_prompt(
                content,
                system_prompt=system_prompt,
                **kwargs,
            )

        # ---------------- mixed or multi-part prompt ---------- 
        elif isinstance(content, list):
            has_image = any(isinstance(item, bytes) for item in content)
            if has_image and not self.is_multimodal:
                raise NotImplementedError(
                    "Multimodal generation is not enabled for this engine."
                )
            return self._generate_from_multiple_input(
                content,
                system_prompt=system_prompt,
                **kwargs,
            )

        else:
            raise ValueError(f"Unsupported input type: {type(content)}")

    # def _generate_from_single_prompt(
    #     self, prompt: str, system_prompt: str=None, temperature=0, max_tokens=2000, top_p=0.99
    # ):

    #     sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

    #     cache_or_none = self._check_cache(sys_prompt_arg + prompt)
    #     if cache_or_none is not None:
    #         return cache_or_none

    #     response = self.client.chat.completions.create(
    #         model=self.model_string,
    #         messages=[
    #             {"role": "system", "content": sys_prompt_arg},
    #             {"role": "user", "content": prompt},
    #         ],
    #         frequency_penalty=0,
    #         presence_penalty=0,
    #         stop=None,
    #         temperature=temperature,
    #         max_tokens=max_tokens,
    #         top_p=top_p,
    #     )

    #     response = response.choices[0].message.content
    #     self._save_cache(sys_prompt_arg + prompt, response)
    #     return response

    def _generate_from_single_prompt(
        self,
        prompt: Union[str, bytes, List[Union[str, bytes]]],
        system_prompt: str | None = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        top_p: float = 0.99,
        response_format=None,
    ):
        sys_prompt_arg = system_prompt or self.system_prompt

        # -------- 1) normalise the prompt into "content" ----------------------
        if isinstance(prompt, (str, bytes)):
            content_list = [prompt]               # promote to list
        elif isinstance(prompt, list):
            content_list = prompt
        else:
            raise ValueError("Prompt must be str, bytes, or list[str|bytes]")

        # If any element is bytes we need a multimodal-capable engine
        if any(isinstance(x, bytes) for x in content_list):
            if not self.is_multimodal:
                raise NotImplementedError("Engine is not multimodal.")

        formatted_content = self._format_content(content_list)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cached = self._check_cache(cache_key)
        if cached is not None:
            if response_format:
                return (
                    cached
                    if isinstance(cached, response_format)
                    else response_format.model_validate_json(cached)
                )
            return cached

        # -------- 2) build messages for the vision/text API -------------------
        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user",   "content": formatted_content},
        ]

        # -------- 3) structured vs plain --------------------------------------
        if response_format:
            resp = self.client.beta.chat.completions.parse(
                model=self.model_string,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            obj = resp.choices[0].message.parsed
            self._save_cache(cache_key, obj.model_dump_json())
            return obj

        # plain-text
        resp = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        text_out = resp.choices[0].message.content
        self._save_cache(cache_key, text_out)
        return text_out



    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API.
        """
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                # For now, bytes are assumed to be images
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt: str | None = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        top_p: float = 0.99,
        response_format=None,          # <-- NEW: Pydantic class or None
    ):
        """
        Accept a mixed list of text strings and image bytes.

        • If `response_format` is None → return raw text  
        • If `response_format` is a Pydantic model → return a parsed instance
        """
        sys_prompt_arg = system_prompt or self.system_prompt

        # Convert images → {"type":"image_url", ...}
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cached = self._check_cache(cache_key)
        if cached is not None:
            if response_format:
                if isinstance(cached, response_format):
                    return cached
                return response_format.model_validate_json(cached)
            return cached

        # -------- build messages --------
        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user",   "content": formatted_content},   # vision payload
        ]

        # -------- structured-output path --------
        if response_format:
            resp = self.client.beta.chat.completions.parse(
                model=self.model_string,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            parsed_obj = resp.choices[0].message.parsed
            # cache JSON text so the cache stays serialisable
            self._save_cache(cache_key, parsed_obj.model_dump_json())
            return parsed_obj

        # -------- plain-text path --------
        resp = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        text_out = resp.choices[0].message.content
        self._save_cache(cache_key, text_out)
        return text_out

class AzureChatOpenAI(ChatOpenAI):
    def __init__(
        self,
        model_string="gpt-35-turbo",
        system_prompt=ChatOpenAI.DEFAULT_SYSTEM_PROMPT,
        **kwargs):
        """
        Initializes an interface for interacting with Azure's OpenAI models.

        This class extends the ChatOpenAI class to use Azure's OpenAI API instead of OpenAI's API. It sets up the necessary client with the appropriate API version, API key, and endpoint from environment variables.

        :param model_string: The model identifier for Azure OpenAI. Defaults to 'gpt-3.5-turbo'.
        :param system_prompt: The default system prompt to use when generating responses. Defaults to ChatOpenAI's default system prompt.
        :param kwargs: Additional keyword arguments to pass to the ChatOpenAI constructor.

        Environment variables:
        - AZURE_OPENAI_API_KEY: The API key for authenticating with Azure OpenAI.
        - AZURE_OPENAI_API_BASE: The base URL for the Azure OpenAI API.
        - AZURE_OPENAI_API_VERSION: The API version to use. Defaults to '2023-07-01-preview' if not set.

        Raises:
            ValueError: If the AZURE_OPENAI_API_KEY environment variable is not set.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_azure_{model_string}.db")  # Changed cache path to differentiate from OpenAI cache

        super().__init__(cache_path=cache_path, system_prompt=system_prompt, **kwargs)

        self.system_prompt = system_prompt
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
        if os.getenv("AZURE_OPENAI_API_KEY") is None:
            raise ValueError("Please set the AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, and AZURE_OPENAI_API_VERSION environment variables if you'd like to use Azure OpenAI models.")
        
        self.client = AzureOpenAI(
            api_version=api_version,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            azure_deployment=model_string,
        )
        self.model_string = model_string
