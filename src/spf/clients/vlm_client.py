import os
import base64
import httpx
import json
from google import genai
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from dotenv import load_dotenv
from typing import Dict, Iterable, List
from pathlib import Path

class VLMClient:
    """
    Unified VLM API client that supports both Gemini and OpenAI providers
    Handles API initialization, configuration, and response processing
    """

    def __init__(self, api_provider: str, model_name: str):
        """
        Initialize VLM client with specified provider and model

        Args:
            api_provider (str): Either "gemini" or "openai"
            model_name (str): Specific model name to use
        """
        self.api_provider = api_provider.lower()
        self.model_name = model_name
        self.openai_client = None
        self.openai_http_client = None
        self.openai_no_auth = False
        self.openai_base_url = None

        # Load environment variables
        env_path = Path(__file__).resolve().parents[3] / ".env"
        load_dotenv(dotenv_path=env_path if env_path.exists() else None, override=False)

        # Initialize the appropriate client
        if self.api_provider == "openai":
            self._init_openai_client()
        elif self.api_provider == "gemini":
            self._init_gemini_client()
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")

    def _init_gemini_client(self):
        """Initialize Gemini client"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=api_key)

        # Store generation config for later use
        self.generation_config = genai.types.GenerateContentConfig(
            temperature=0.4,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )

    def _init_openai_client(self):
        """Initialize OpenAI client with configurable base URL"""
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
        self.openai_base_url = base_url.rstrip("/")

        is_private_openai_base = self.openai_base_url.startswith(
            ("http://127.0.0.1", "http://localhost", "http://10.")
        )
        placeholder_keys = {"", "dummy", "empty", "none", "null", "your_openai_api_key_here"}
        normalized_key = (api_key or "").strip().lower()

        if is_private_openai_base and normalized_key in placeholder_keys:
            self.openai_no_auth = True
            self.openai_http_client = httpx.Client(trust_env=False, timeout=60.0)
            print(
                f"[VLMClient] Using no-auth OpenAI-compatible mode for {self.openai_base_url}"
            )
            return

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        client_kwargs = {
            "api_key": api_key,
            "base_url": self.openai_base_url,
        }

        if is_private_openai_base:
            client_kwargs["http_client"] = httpx.Client(trust_env=False)

        self.openai_client = OpenAI(**client_kwargs)

    def generate_response(self, prompt: str, image) -> str:
        """
        Generate response from VLM with image and text prompt

        Args:
            prompt (str): Text prompt for the model
            image: Image data (numpy array)

        Returns:
            str: Raw response text from the model
        """
        # Encode image to base64
        import cv2

        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        if self.api_provider == "openai":
            return self._get_openai_response(prompt, encoded_image)
        else:
            return self._get_gemini_response(prompt, encoded_image)

    def _get_gemini_response(self, prompt: str, encoded_image: str) -> str:
        """Get response from Gemini API"""
        import base64
        import io
        from PIL import Image

        # Decode base64 string to bytes and convert to PIL Image
        image_bytes = base64.b64decode(encoded_image)
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Use PIL Image directly as the migration guide suggests
        contents = [prompt, pil_image]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self.generation_config
        )
        return response.text or ""

    def _get_openai_response(self, prompt: str, encoded_image: str) -> str:
        """Get response from OpenAI API with image"""
        messages: List[ChatCompletionUserMessageParam] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
        extra_body = None
        if "qwen3" in self.model_name.lower():
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        if self.openai_no_auth:
            request_body = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 512,
            }
            if extra_body:
                request_body.update(extra_body)
            response = self.openai_http_client.post(
                f"{self.openai_base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://spf-web.pages.dev",
                    "X-Title": "See, Point, Fly",
                },
                json=request_body,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"] or ""

        request_kwargs = {
            "extra_headers": {
                "HTTP-Referer": "https://spf-web.pages.dev",
                "X-Title": "See, Point, Fly"
            },
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 512,
            "timeout": 60,
        }
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        response = self.openai_client.chat.completions.create(
            **request_kwargs
        )

        return response.choices[0].message.content or ""

    @staticmethod
    def clean_response_text(response_text: str) -> str:
        """Clean response text from markdown formatting"""
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        return response_text

    @staticmethod
    def parse_candidate_selection(
        response_text: str,
        valid_choices: Iterable[str],
        fallback_choice: str = "P8",
    ) -> Dict[str, object]:
        """Parse SPF-IEVE-Lite candidate JSON with strict fallback behavior."""
        fallback = {
            "choice": fallback_choice,
            "target_visible": False,
            "confidence": 0.0,
            "reason": f"fallback to {fallback_choice}",
            "fallback_used": True,
        }

        try:
            cleaned_text = VLMClient.clean_response_text(response_text.strip())
            response_data = json.loads(cleaned_text)
            if not isinstance(response_data, dict):
                return fallback

            required_fields = ("choice", "target_visible", "confidence", "reason")
            if any(field not in response_data for field in required_fields):
                return fallback

            valid_choice_set = {str(choice).upper() for choice in valid_choices}
            choice = str(response_data["choice"]).strip().upper()
            if choice not in valid_choice_set:
                return fallback

            confidence = float(response_data["confidence"])
            confidence = max(0.0, min(1.0, confidence))

            return {
                "choice": choice,
                "target_visible": _parse_bool(response_data["target_visible"]),
                "confidence": confidence,
                "reason": str(response_data["reason"]),
                "fallback_used": False,
            }
        except Exception:
            return fallback


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)
