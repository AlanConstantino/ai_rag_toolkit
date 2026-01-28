"""Tests for the security module."""

import unittest

from rag_system.security import (
    validate_query_length,
    validate_query_length_or_raise,
    validate_content_length,
    sanitize_html_content,
    sanitize_for_logging,
    check_for_suspicious_content,
    ValidationError,
    MAX_QUERY_LENGTH,
    MAX_CONTENT_LENGTH,
)


class TestQueryLengthValidation(unittest.TestCase):
    """Tests for query length validation."""

    def test_valid_query(self):
        """validate_query_length should accept valid queries."""
        is_valid, error = validate_query_length("What is Python?")

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_empty_query_invalid(self):
        """validate_query_length should reject empty queries."""
        is_valid, error = validate_query_length("")

        self.assertFalse(is_valid)
        self.assertIn("empty", error.lower())

    def test_too_long_query_invalid(self):
        """validate_query_length should reject queries that are too long."""
        long_query = "a" * (MAX_QUERY_LENGTH + 1)

        is_valid, error = validate_query_length(long_query)

        self.assertFalse(is_valid)
        self.assertIn("exceeds", error.lower())

    def test_max_length_query_valid(self):
        """validate_query_length should accept queries at max length."""
        max_query = "a" * MAX_QUERY_LENGTH

        is_valid, error = validate_query_length(max_query)

        self.assertTrue(is_valid)

    def test_custom_max_length(self):
        """validate_query_length should respect custom max_length."""
        is_valid, error = validate_query_length("hello world", max_length=5)

        self.assertFalse(is_valid)
        self.assertIn("5", error)

    def test_or_raise_raises_on_invalid(self):
        """validate_query_length_or_raise should raise ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            validate_query_length_or_raise("")

        self.assertEqual(ctx.exception.field, "query")


class TestContentLengthValidation(unittest.TestCase):
    """Tests for content length validation."""

    def test_valid_content(self):
        """validate_content_length should accept valid content."""
        is_valid, error = validate_content_length("Hello world")

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_none_content_valid(self):
        """validate_content_length should accept None content."""
        is_valid, error = validate_content_length(None)

        self.assertTrue(is_valid)

    def test_too_long_content_invalid(self):
        """validate_content_length should reject content that is too long."""
        long_content = "a" * (MAX_CONTENT_LENGTH + 1)

        is_valid, error = validate_content_length(long_content)

        self.assertFalse(is_valid)
        self.assertIn("exceeds", error.lower())


class TestHTMLSanitization(unittest.TestCase):
    """Tests for HTML content sanitization."""

    def test_removes_script_tags(self):
        """sanitize_html_content should remove script tags."""
        html = '<html><body><script>alert("xss")</script>Hello</body></html>'

        result = sanitize_html_content(html)

        self.assertNotIn("<script>", result)
        self.assertNotIn("alert", result)
        self.assertIn("Hello", result)

    def test_removes_style_tags(self):
        """sanitize_html_content should remove style tags."""
        html = '<html><body><style>body{color:red}</style>Hello</body></html>'

        result = sanitize_html_content(html)

        self.assertNotIn("<style>", result)
        self.assertNotIn("color:red", result)
        self.assertIn("Hello", result)

    def test_removes_event_handlers(self):
        """sanitize_html_content should remove event handlers."""
        html = '<a href="#" onclick="alert(1)">Click me</a>'

        result = sanitize_html_content(html)

        self.assertNotIn("onclick", result)
        self.assertIn("Click me", result)

    def test_removes_javascript_protocol(self):
        """sanitize_html_content should remove javascript: protocol."""
        html = '<a href="javascript:alert(1)">Click me</a>'

        result = sanitize_html_content(html)

        self.assertNotIn("javascript:", result)

    def test_removes_data_urls(self):
        """sanitize_html_content should remove data:text/html URLs."""
        html = '<iframe src="data:text/html,<script>alert(1)</script>"></iframe>'

        result = sanitize_html_content(html)

        self.assertNotIn("data:text/html", result)

    def test_preserves_safe_content(self):
        """sanitize_html_content should preserve safe content."""
        html = '<html><body><h1>Title</h1><p>Paragraph</p></body></html>'

        result = sanitize_html_content(html)

        self.assertIn("<h1>Title</h1>", result)
        self.assertIn("<p>Paragraph</p>", result)

    def test_handles_empty_input(self):
        """sanitize_html_content should handle empty input."""
        self.assertEqual(sanitize_html_content(""), "")
        self.assertIsNone(sanitize_html_content(None))


class TestLoggingSanitization(unittest.TestCase):
    """Tests for logging sanitization."""

    def test_truncates_long_values(self):
        """sanitize_for_logging should truncate long values."""
        long_value = "a" * 500

        result = sanitize_for_logging(long_value, max_length=100)

        self.assertTrue(len(result) < 150)  # Truncated + marker
        self.assertIn("[truncated]", result)

    def test_redacts_password_patterns(self):
        """sanitize_for_logging should redact password patterns."""
        value = "config with password=secret123 included"

        result = sanitize_for_logging(value)

        self.assertNotIn("secret123", result)
        self.assertIn("[REDACTED]", result)

    def test_redacts_token_patterns(self):
        """sanitize_for_logging should redact token patterns."""
        value = "Authorization: token=abc123xyz"

        result = sanitize_for_logging(value)

        self.assertIn("[REDACTED]", result)

    def test_redacts_long_api_keys(self):
        """sanitize_for_logging should redact long alphanumeric strings."""
        # Use a generic pattern that looks like an API key (32+ alphanumeric)
        value = "API key: aaaa1111bbbb2222cccc3333dddd4444eeee5555"

        result = sanitize_for_logging(value)

        self.assertIn("[REDACTED]", result)
        self.assertNotIn("aaaa1111bbbb", result)

    def test_handles_empty_input(self):
        """sanitize_for_logging should handle empty input."""
        self.assertEqual(sanitize_for_logging(""), "")
        self.assertIsNone(sanitize_for_logging(None))


class TestSuspiciousContentCheck(unittest.TestCase):
    """Tests for suspicious content detection."""

    def test_detects_script_tags(self):
        """check_for_suspicious_content should detect script tags."""
        content = '<script>alert("xss")</script>'

        has_suspicious, description = check_for_suspicious_content(content)

        self.assertTrue(has_suspicious)
        self.assertIsNotNone(description)

    def test_detects_event_handlers(self):
        """check_for_suspicious_content should detect event handlers."""
        content = '<img src="x" onerror="alert(1)">'

        has_suspicious, description = check_for_suspicious_content(content)

        self.assertTrue(has_suspicious)

    def test_detects_javascript_protocol(self):
        """check_for_suspicious_content should detect javascript: protocol."""
        content = '<a href="javascript:void(0)">Link</a>'

        has_suspicious, description = check_for_suspicious_content(content)

        self.assertTrue(has_suspicious)

    def test_safe_content_not_flagged(self):
        """check_for_suspicious_content should not flag safe content."""
        content = '<html><body><p>Hello world</p></body></html>'

        has_suspicious, description = check_for_suspicious_content(content)

        self.assertFalse(has_suspicious)
        self.assertIsNone(description)

    def test_handles_empty_input(self):
        """check_for_suspicious_content should handle empty input."""
        has_suspicious, description = check_for_suspicious_content("")

        self.assertFalse(has_suspicious)
        self.assertIsNone(description)


class TestValidationError(unittest.TestCase):
    """Tests for ValidationError exception."""

    def test_stores_field_name(self):
        """ValidationError should store field name."""
        error = ValidationError("username", "too short")

        self.assertEqual(error.field, "username")
        self.assertIn("username", str(error))
        self.assertIn("too short", str(error))


if __name__ == '__main__':
    unittest.main()
