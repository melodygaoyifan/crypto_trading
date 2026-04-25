"""
Regression tests for P24 (Discord webhook circuit breaker + Haiku 429 path).

Discord:
- Permanent disable on HTTP 401/403/404 (revoked webhook URL)
- Cooldown on 429 honoring Retry-After header (capped 1-300s)
- Exponential cooldown after 5/10/15 consecutive transient failures
- Reset on first successful post

Haiku:
- 429 reclassified from "transient_error" to non-retryable with short cooldown
"""
import sys, os
import time
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infra.persistence import DiscordNotifier
from agents.sentiment_llm_agent import LLMSentimentAgent


# =============================================================================
# Discord circuit breaker
# =============================================================================

def _make_notifier(url="https://example.invalid/webhook"):
    n = DiscordNotifier(webhook_url=url, enable_async=False)
    return n


class TestDiscordCircuitBreaker:
    def test_starts_closed_no_failures(self):
        n = _make_notifier()
        assert n._consecutive_failures == 0
        assert n._circuit_open_until == 0.0
        assert n._circuit_permanently_disabled is False

    def test_401_permanently_disables(self):
        import urllib.error
        n = _make_notifier()
        err = urllib.error.HTTPError(
            url=n.webhook_url, code=401, msg="Unauthorized",
            hdrs=None, fp=MagicMock(read=lambda: b"unauthorized"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            n._post_webhook({"content": "test"})
        assert n._circuit_permanently_disabled is True

    def test_403_permanently_disables(self):
        import urllib.error
        n = _make_notifier()
        err = urllib.error.HTTPError(
            url=n.webhook_url, code=403, msg="Forbidden",
            hdrs=None, fp=MagicMock(read=lambda: b"forbidden"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            n._post_webhook({"content": "test"})
        assert n._circuit_permanently_disabled is True

    def test_429_opens_short_circuit_with_retry_after(self):
        import urllib.error
        n = _make_notifier()
        # Mock headers with Retry-After
        hdrs = MagicMock()
        hdrs.get = lambda k, d=None: "5" if k == "Retry-After" else d
        err = urllib.error.HTTPError(
            url=n.webhook_url, code=429, msg="Too Many Requests",
            hdrs=hdrs, fp=MagicMock(read=lambda: b"rate limited"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            n._post_webhook({"content": "test"})
        # Circuit open for ~5 seconds (Retry-After value)
        assert n._circuit_open_until > time.time()
        assert n._circuit_open_until - time.time() <= 6.0
        assert n._circuit_permanently_disabled is False

    def test_429_caps_retry_after_at_300s(self):
        import urllib.error
        n = _make_notifier()
        hdrs = MagicMock()
        hdrs.get = lambda k, d=None: "99999" if k == "Retry-After" else d
        err = urllib.error.HTTPError(
            url=n.webhook_url, code=429, msg="Too Many Requests",
            hdrs=hdrs, fp=MagicMock(read=lambda: b"rate limited"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            n._post_webhook({"content": "test"})
        # Capped at 300s
        assert n._circuit_open_until - time.time() <= 301.0

    def test_5_transient_failures_open_circuit(self):
        import urllib.error
        n = _make_notifier()
        err = urllib.error.HTTPError(
            url=n.webhook_url, code=500, msg="Internal Server Error",
            hdrs=None, fp=MagicMock(read=lambda: b"oops"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            for _ in range(5):
                n._post_webhook({"content": "test"})
        assert n._consecutive_failures == 5
        assert n._circuit_open_until > time.time()

    def test_4_failures_no_circuit_yet(self):
        import urllib.error
        n = _make_notifier()
        err = urllib.error.HTTPError(
            url=n.webhook_url, code=500, msg="Internal Server Error",
            hdrs=None, fp=MagicMock(read=lambda: b"oops"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            for _ in range(4):
                n._post_webhook({"content": "test"})
        assert n._consecutive_failures == 4
        # 4 < 5 threshold → circuit not yet open
        assert n._circuit_open_until == 0.0

    def test_open_circuit_drops_messages_silently(self):
        n = _make_notifier()
        n._circuit_open_until = time.time() + 60.0
        with patch("urllib.request.urlopen") as mock_urlopen:
            n._post_webhook({"content": "test"})
            assert mock_urlopen.call_count == 0  # never reached the network

    def test_permanent_disable_drops_messages(self):
        n = _make_notifier()
        n._circuit_permanently_disabled = True
        with patch("urllib.request.urlopen") as mock_urlopen:
            n._post_webhook({"content": "test"})
            assert mock_urlopen.call_count == 0


# =============================================================================
# Haiku 429 classification
# =============================================================================

class TestHaiku429Classification:
    def test_429_status_code_classified_as_rate_limit(self):
        err = Exception("rate limit hit")
        err.status_code = 429
        status, non_retryable, reason = LLMSentimentAgent._classify_haiku_error(err)
        assert status == 429
        assert non_retryable is True  # treat as non-retryable to trigger cooldown
        assert "rate_limit_429" in reason

    def test_429_in_message_classified(self):
        err = Exception("HTTP 429: too many requests")
        status, non_retryable, reason = LLMSentimentAgent._classify_haiku_error(err)
        assert status == 429
        assert "rate_limit_429" in reason

    def test_rate_limit_text_classified(self):
        err = Exception("rate limit exceeded")
        _, non_retryable, reason = LLMSentimentAgent._classify_haiku_error(err)
        assert non_retryable is True
        assert "rate_limit_429" in reason

    def test_429_with_retry_after_header_parsed(self):
        err = Exception("rate limited")
        err.status_code = 429
        # Mock response with Retry-After header
        resp = MagicMock()
        resp.headers = {"retry-after": "12"}
        err.response = resp
        _, _, reason = LLMSentimentAgent._classify_haiku_error(err)
        assert "retry_after=12" in reason

    def test_403_still_classified_separately(self):
        err = Exception("forbidden")
        err.status_code = 403
        status, non_retryable, reason = LLMSentimentAgent._classify_haiku_error(err)
        assert status == 403
        assert non_retryable is True
        assert "rate_limit" not in reason  # NOT a rate limit

    def test_500_still_transient(self):
        err = Exception("internal error")
        err.status_code = 500
        _, non_retryable, reason = LLMSentimentAgent._classify_haiku_error(err)
        assert non_retryable is False
        assert reason == "transient_error"
