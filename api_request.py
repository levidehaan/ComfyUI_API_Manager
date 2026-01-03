"""
API Request Node

Enhanced node for making API requests with support for:
- Multiple authentication methods (Bearer, API Key, Basic, OAuth2)
- GET/POST/PUT/DELETE methods
- Response parsing and error handling
- Timeout and retry configuration
"""

import json
import logging
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class APIRequestNode:
    """
    A comprehensive node for making API requests and processing responses.

    Supports multiple HTTP methods, authentication types, and robust error handling.
    """

    # Authentication types
    AUTH_TYPES = ["none", "bearer", "api_key", "basic", "oauth2"]
    HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "https://api.example.com/endpoint"
                }),
                "method": (cls.HTTP_METHODS, {"default": "GET"}),
                "auth_type": (cls.AUTH_TYPES, {"default": "none"}),
            },
            "optional": {
                "auth_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "OAuth2 token endpoint (if needed)"
                }),
                "auth_credentials": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "JSON credentials or token"
                }),
                "request_body": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "JSON request body (for POST/PUT)"
                }),
                "headers": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Additional headers as JSON"
                }),
                "query_params": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Query parameters as JSON"
                }),
                "response_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Dot notation path to extract (e.g., data.items)"
                }),
                "array_index": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
                "timeout_seconds": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 300.0,
                    "step": 1.0
                }),
                "retry_count": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("JSON", "INT", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("RESPONSE", "LENGTH", "AUTH_TOKEN", "SUCCESS", "ERROR")
    FUNCTION = "execute"
    CATEGORY = "API Manager"

    def __init__(self):
        self._session: Optional[requests.Session] = None
        self._cached_token: Optional[str] = None
        self._token_expiry: Optional[float] = None

    def _get_session(self, retry_count: int = 3) -> requests.Session:
        """Get or create a requests session with retry logic."""
        if self._session is None:
            self._session = requests.Session()

            retry_strategy = Retry(
                total=retry_count,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "PATCH"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

        return self._session

    def _parse_json_input(self, text: str) -> Optional[dict]:
        """Safely parse JSON input, handling common issues."""
        if not text or not text.strip():
            return None

        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try wrapping with braces
        try:
            return json.loads("{" + text + "}")
        except json.JSONDecodeError:
            pass

        logger.warning(f"Could not parse JSON: {text[:100]}...")
        return None

    def _extract_path(self, data: Any, path: str) -> Any:
        """Extract data using dot notation path."""
        if not path:
            return data

        parts = path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx] if 0 <= idx < len(current) else None
                except (ValueError, IndexError):
                    current = None
            else:
                return None

            if current is None:
                return None

        return current

    def authenticate(
        self,
        auth_type: str,
        auth_url: str,
        credentials: dict,
        session: requests.Session,
        timeout: float
    ) -> Optional[str]:
        """
        Authenticate and retrieve a token.

        Args:
            auth_type: Type of authentication
            auth_url: Authentication endpoint
            credentials: Authentication credentials
            session: Requests session
            timeout: Request timeout

        Returns:
            Authentication token or None
        """
        if auth_type == "none":
            return None

        if auth_type == "bearer":
            # Credentials should contain the token directly
            return credentials.get("token") or credentials.get("access_token")

        if auth_type == "api_key":
            return credentials.get("api_key") or credentials.get("key")

        if auth_type == "basic":
            # Return base64 encoded credentials
            import base64
            username = credentials.get("username", "")
            password = credentials.get("password", "")
            return base64.b64encode(f"{username}:{password}".encode()).decode()

        if auth_type == "oauth2":
            # Perform OAuth2 token request
            try:
                token_data = {
                    "grant_type": credentials.get("grant_type", "client_credentials"),
                    "client_id": credentials.get("client_id"),
                    "client_secret": credentials.get("client_secret"),
                }

                if credentials.get("scope"):
                    token_data["scope"] = credentials["scope"]

                response = session.post(
                    auth_url,
                    data=token_data,
                    timeout=timeout
                )
                response.raise_for_status()

                token_response = response.json()
                return token_response.get("access_token")

            except Exception as e:
                logger.error(f"OAuth2 authentication failed: {e}")
                return None

        return None

    def _build_headers(
        self,
        auth_type: str,
        token: Optional[str],
        extra_headers: Optional[dict]
    ) -> dict:
        """Build request headers including authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        if token:
            if auth_type == "bearer" or auth_type == "oauth2":
                headers["Authorization"] = f"Bearer {token}"
            elif auth_type == "api_key":
                headers["X-API-Key"] = token
            elif auth_type == "basic":
                headers["Authorization"] = f"Basic {token}"

        if extra_headers:
            headers.update(extra_headers)

        return headers

    def execute(
        self,
        api_url: str,
        method: str = "GET",
        auth_type: str = "none",
        auth_url: str = "",
        auth_credentials: str = "",
        request_body: str = "",
        headers: str = "",
        query_params: str = "",
        response_path: str = "",
        array_index: int = 0,
        timeout_seconds: float = 30.0,
        retry_count: int = 3
    ):
        """
        Execute the API request.

        Returns:
            Tuple of (response_data, length, auth_token, success, error_message)
        """
        try:
            session = self._get_session(retry_count)

            # Parse inputs
            credentials = self._parse_json_input(auth_credentials) or {}
            body = self._parse_json_input(request_body)
            extra_headers = self._parse_json_input(headers) or {}
            params = self._parse_json_input(query_params) or {}

            # Authenticate if needed
            token = self.authenticate(
                auth_type, auth_url, credentials, session, timeout_seconds
            )

            # Build headers
            request_headers = self._build_headers(auth_type, token, extra_headers)

            # Make request
            method = method.upper()
            response = session.request(
                method=method,
                url=api_url,
                headers=request_headers,
                json=body if method in ["POST", "PUT", "PATCH"] and body else None,
                params=params,
                timeout=timeout_seconds
            )

            response.raise_for_status()

            # Parse response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"raw_response": response.text}

            # Extract path if specified
            if response_path:
                response_data = self._extract_path(response_data, response_path)

            # Handle array index
            length = 0
            if isinstance(response_data, list):
                length = len(response_data)
                if array_index >= 0 and array_index < length:
                    response_data = response_data[array_index]
                elif array_index == -1:
                    # -1 means return the full array
                    pass
                else:
                    response_data = {}

            auth_token_str = f"Bearer {token}" if token else ""

            logger.info(f"API request successful: {method} {api_url}")
            return (response_data or {}, length, auth_token_str, True, "")

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {timeout_seconds}s"
            logger.error(error_msg)
            return ({}, 0, "", False, error_msg)

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {e}"
            logger.error(error_msg)
            return ({}, 0, "", False, error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
            logger.error(error_msg)
            return ({}, 0, "", False, error_msg)

        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return ({}, 0, "", False, error_msg)


class APIRequestNodeSimple:
    """
    Simplified API request node for quick GET requests.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            },
            "optional": {
                "bearer_token": ("STRING", {"default": "", "forceInput": True}),
                "response_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("JSON", "BOOLEAN")
    RETURN_NAMES = ("RESPONSE", "SUCCESS")
    FUNCTION = "execute"
    CATEGORY = "API Manager"

    def execute(self, url: str, bearer_token: str = "", response_path: str = ""):
        try:
            headers = {"Accept": "application/json"}
            if bearer_token:
                if not bearer_token.startswith("Bearer "):
                    bearer_token = f"Bearer {bearer_token}"
                headers["Authorization"] = bearer_token

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if response_path:
                parts = response_path.split(".")
                for part in parts:
                    if isinstance(data, dict):
                        data = data.get(part, {})
                    elif isinstance(data, list):
                        try:
                            data = data[int(part)]
                        except (ValueError, IndexError):
                            data = {}

            return (data, True)

        except Exception as e:
            logger.error(f"Simple API request failed: {e}")
            return ({}, False)
