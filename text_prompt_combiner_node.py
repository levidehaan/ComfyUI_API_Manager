"""
Text Prompt Combiner Node

Combines API response data with text templates for dynamic prompt generation.
Supports template substitution, JSON path extraction, and CLIP encoding.
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TextPromptCombinerNode:
    """
    Combines API response data with a text template to create dynamic prompts.

    Supports:
    - $variable substitution from API response
    - Nested path access (e.g., $data.items.0.name)
    - Optional CLIP encoding for use in image generation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt_template": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "A $style portrait of $subject with $details"
                }),
            },
            "optional": {
                "api_response": ("JSON", {"default": {}}),
                "id_field_name": ("STRING", {
                    "default": "",
                    "placeholder": "Field name to extract as ID (e.g., 'id' or 'uuid')"
                }),
                "clip": ("CLIP",),
                "default_values": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Default values as JSON (used when API response is missing keys)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("TEXT", "CONDITIONING", "ID")
    FUNCTION = "execute"
    CATEGORY = "API Manager"

    def _extract_nested_value(self, data: Any, path: str) -> Any:
        """
        Extract a value from nested data using dot notation.

        Args:
            data: The data to extract from
            path: Dot-separated path (e.g., "user.profile.name" or "items.0.title")

        Returns:
            The extracted value or None if not found
        """
        parts = path.split(".")
        current = data

        for part in parts:
            if current is None:
                return None

            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx] if 0 <= idx < len(current) else None
                except ValueError:
                    return None
            else:
                return None

        return current

    def _substitute_template(
        self,
        template: str,
        data: dict,
        defaults: Optional[dict] = None
    ) -> str:
        """
        Substitute $variable placeholders in template with values from data.

        Supports:
        - Simple: $name -> data["name"]
        - Nested: $user.name -> data["user"]["name"]
        - Array: $items.0.title -> data["items"][0]["title"]
        """
        if not isinstance(data, dict):
            return template

        defaults = defaults or {}

        # Find all $variable patterns (including nested paths)
        pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)*)'

        def replace_match(match):
            path = match.group(1)

            # Try to get value from data
            value = self._extract_nested_value(data, path)

            # Fall back to defaults if not found
            if value is None:
                value = defaults.get(path)

            # If still None, try simple key lookup
            if value is None and "." not in path:
                value = data.get(path) or defaults.get(path)

            # Convert to string
            if value is not None:
                return str(value)
            else:
                # Keep original placeholder if not found
                logger.warning(f"No value found for placeholder: ${path}")
                return match.group(0)

        return re.sub(pattern, replace_match, template)

    def execute(
        self,
        text_prompt_template: str,
        api_response: Optional[dict] = None,
        id_field_name: str = "",
        clip=None,
        default_values: str = ""
    ):
        """
        Execute the template substitution and optional CLIP encoding.

        Returns:
            Tuple of (combined_text, conditioning, extracted_id)
        """
        api_response = api_response or {}

        # Parse default values
        defaults = {}
        if default_values and default_values.strip():
            try:
                defaults = json.loads(default_values)
            except json.JSONDecodeError:
                logger.warning("Could not parse default_values as JSON")

        # Substitute template
        combined_text = self._substitute_template(
            text_prompt_template,
            api_response,
            defaults
        )

        logger.info(f"Combined text prompt: {combined_text[:100]}...")

        # Extract ID if field name provided
        extracted_id = ""
        if id_field_name:
            extracted_id = str(self._extract_nested_value(api_response, id_field_name) or "")
            logger.info(f"Extracted ID: {extracted_id}")

        # Encode with CLIP if provided
        conditioning = None
        if clip is not None:
            try:
                tokens = clip.tokenize(combined_text)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                conditioning = [[cond, {"pooled_output": pooled}]]
            except Exception as e:
                logger.error(f"CLIP encoding failed: {e}")
                conditioning = None

        return (combined_text, conditioning, extracted_id)


class TextTemplateNode:
    """
    Simple text template node without CLIP encoding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Hello, $name! Your order #$order_id is ready."
                }),
            },
            "optional": {
                "variables": ("JSON", {"default": {}}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TEXT",)
    FUNCTION = "execute"
    CATEGORY = "API Manager"

    def execute(self, template: str, variables: Optional[dict] = None):
        variables = variables or {}

        # Simple substitution
        result = template
        if isinstance(variables, dict):
            for key, value in variables.items():
                placeholder = f"${key}"
                result = result.replace(placeholder, str(value))

        return (result,)


class JSONExtractNode:
    """
    Extracts values from JSON data using path notation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": ("JSON", {"default": {}}),
                "path": ("STRING", {
                    "default": "",
                    "placeholder": "data.items.0.name"
                }),
            },
            "optional": {
                "default_value": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "JSON", "BOOLEAN")
    RETURN_NAMES = ("VALUE_STRING", "VALUE_JSON", "FOUND")
    FUNCTION = "extract"
    CATEGORY = "API Manager"

    def extract(
        self,
        json_data: Any,
        path: str,
        default_value: str = ""
    ):
        if not path:
            return (str(json_data), json_data, True)

        parts = path.split(".")
        current = json_data

        for part in parts:
            if current is None:
                return (default_value, {}, False)

            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx] if 0 <= idx < len(current) else None
                except ValueError:
                    return (default_value, {}, False)
            else:
                return (default_value, {}, False)

        if current is None:
            return (default_value, {}, False)

        return (str(current), current, True)
