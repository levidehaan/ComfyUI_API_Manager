"""
JSON Array Iterator Node

Provides utilities for working with JSON arrays including:
- Array iteration and selection
- Array manipulation (filter, map, slice)
- Array statistics
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JSONArrayIteratorNode:
    """
    Iterates through a JSON array and outputs specific objects
    based on a selection index.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_array": ("JSON", {"default": []}),
                "selection_index": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("JSON", "INT", "BOOLEAN")
    RETURN_NAMES = ("SELECTED_ITEM", "ARRAY_LENGTH", "HAS_MORE")
    FUNCTION = "iterate"
    CATEGORY = "API Manager"

    def iterate(self, json_array: Any, selection_index: int = 0):
        """
        Select an item from the array.

        Args:
            json_array: The array to iterate
            selection_index: Index to select (-1 for last item)

        Returns:
            Tuple of (selected_item, array_length, has_more_items)
        """
        if not isinstance(json_array, list):
            logger.warning("Input is not a list, returning empty")
            return ({}, 0, False)

        length = len(json_array)

        if length == 0:
            return ({}, 0, False)

        # Handle negative index (Python-style)
        if selection_index < 0:
            selection_index = length + selection_index

        # Bounds check
        if selection_index < 0 or selection_index >= length:
            logger.warning(f"Index {selection_index} out of bounds for array of length {length}")
            return ({}, length, False)

        selected = json_array[selection_index]
        has_more = selection_index < length - 1

        logger.debug(f"Selected item at index {selection_index} of {length}")
        return (selected, length, has_more)


class JSONArrayFilterNode:
    """
    Filters a JSON array based on a field value.
    """

    OPERATORS = ["equals", "not_equals", "contains", "not_contains", "greater", "less", "exists"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_array": ("JSON", {"default": []}),
                "field_path": ("STRING", {
                    "default": "",
                    "placeholder": "status or user.role"
                }),
                "operator": (cls.OPERATORS, {"default": "equals"}),
            },
            "optional": {
                "value": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("JSON", "INT")
    RETURN_NAMES = ("FILTERED_ARRAY", "COUNT")
    FUNCTION = "filter"
    CATEGORY = "API Manager"

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get a value from a nested object using dot notation."""
        if not path:
            return obj

        parts = path.split(".")
        current = obj

        for part in parts:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return current

    def filter(
        self,
        json_array: Any,
        field_path: str,
        operator: str = "equals",
        value: str = ""
    ):
        if not isinstance(json_array, list):
            return ([], 0)

        filtered = []

        for item in json_array:
            field_value = self._get_nested_value(item, field_path)

            match = False

            if operator == "equals":
                match = str(field_value) == value
            elif operator == "not_equals":
                match = str(field_value) != value
            elif operator == "contains":
                match = value.lower() in str(field_value).lower()
            elif operator == "not_contains":
                match = value.lower() not in str(field_value).lower()
            elif operator == "greater":
                try:
                    match = float(field_value) > float(value)
                except (ValueError, TypeError):
                    match = False
            elif operator == "less":
                try:
                    match = float(field_value) < float(value)
                except (ValueError, TypeError):
                    match = False
            elif operator == "exists":
                match = field_value is not None

            if match:
                filtered.append(item)

        return (filtered, len(filtered))


class JSONArraySliceNode:
    """
    Slices a JSON array by start and end indices.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_array": ("JSON", {"default": []}),
            },
            "optional": {
                "start": ("INT", {
                    "default": 0,
                    "min": -9999,
                    "max": 9999
                }),
                "end": ("INT", {
                    "default": -1,
                    "min": -9999,
                    "max": 9999
                }),
                "step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100
                }),
            }
        }

    RETURN_TYPES = ("JSON", "INT")
    RETURN_NAMES = ("SLICED_ARRAY", "COUNT")
    FUNCTION = "slice"
    CATEGORY = "API Manager"

    def slice(
        self,
        json_array: Any,
        start: int = 0,
        end: int = -1,
        step: int = 1
    ):
        if not isinstance(json_array, list):
            return ([], 0)

        # Handle -1 as "to the end"
        if end == -1:
            end = None

        sliced = json_array[start:end:step]
        return (sliced, len(sliced))


class JSONArrayMapNode:
    """
    Extracts a specific field from each item in an array.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_array": ("JSON", {"default": []}),
                "field_path": ("STRING", {
                    "default": "",
                    "placeholder": "name or user.email"
                }),
            }
        }

    RETURN_TYPES = ("JSON", "STRING")
    RETURN_NAMES = ("MAPPED_ARRAY", "AS_STRING")
    FUNCTION = "map_field"
    CATEGORY = "API Manager"

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        if not path:
            return obj

        parts = path.split(".")
        current = obj

        for part in parts:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return current

    def map_field(self, json_array: Any, field_path: str):
        if not isinstance(json_array, list):
            return ([], "")

        mapped = []
        for item in json_array:
            value = self._get_nested_value(item, field_path)
            if value is not None:
                mapped.append(value)

        # Create string representation
        as_string = ", ".join(str(v) for v in mapped)

        return (mapped, as_string)


class JSONArrayStatsNode:
    """
    Calculates statistics for a JSON array.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_array": ("JSON", {"default": []}),
            },
            "optional": {
                "numeric_field": ("STRING", {
                    "default": "",
                    "placeholder": "price or score"
                }),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("COUNT", "SUM", "AVERAGE", "MIN", "MAX")
    FUNCTION = "calculate"
    CATEGORY = "API Manager"

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        if not path:
            return obj

        parts = path.split(".")
        current = obj

        for part in parts:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return current

    def calculate(self, json_array: Any, numeric_field: str = ""):
        if not isinstance(json_array, list):
            return (0, 0.0, 0.0, 0.0, 0.0)

        count = len(json_array)

        if count == 0:
            return (0, 0.0, 0.0, 0.0, 0.0)

        # If no numeric field specified, just return count
        if not numeric_field:
            return (count, 0.0, 0.0, 0.0, 0.0)

        # Extract numeric values
        values = []
        for item in json_array:
            value = self._get_nested_value(item, numeric_field)
            try:
                values.append(float(value))
            except (ValueError, TypeError):
                continue

        if not values:
            return (count, 0.0, 0.0, 0.0, 0.0)

        total = sum(values)
        average = total / len(values)
        min_val = min(values)
        max_val = max(values)

        return (count, total, average, min_val, max_val)


class JSONMergeNode:
    """
    Merges multiple JSON objects or arrays.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_a": ("JSON", {"default": {}}),
            },
            "optional": {
                "json_b": ("JSON", {"default": {}}),
                "json_c": ("JSON", {"default": {}}),
            }
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("MERGED",)
    FUNCTION = "merge"
    CATEGORY = "API Manager"

    def merge(self, json_a: Any, json_b: Any = None, json_c: Any = None):
        # Handle arrays
        if isinstance(json_a, list):
            result = list(json_a)
            if isinstance(json_b, list):
                result.extend(json_b)
            if isinstance(json_c, list):
                result.extend(json_c)
            return (result,)

        # Handle objects
        if isinstance(json_a, dict):
            result = dict(json_a)
            if isinstance(json_b, dict):
                result.update(json_b)
            if isinstance(json_c, dict):
                result.update(json_c)
            return (result,)

        return (json_a,)
