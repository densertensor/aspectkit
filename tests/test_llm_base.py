"""Tests for the connector base: message helpers and CallableChat."""

from aspectkit.llm.base import CallableChat, split_system


class TestSplitSystem:
    def test_no_system(self):
        messages = [{"role": "user", "content": "hi"}]
        system, rest = split_system(messages)
        assert system is None
        assert rest == messages

    def test_single_system(self):
        system, rest = split_system(
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
        )
        assert system == "be brief"
        assert rest == [{"role": "user", "content": "hi"}]

    def test_multiple_system_joined_in_order(self):
        system, _ = split_system(
            [
                {"role": "system", "content": "one"},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "two"},
            ]
        )
        assert system == "one\n\ntwo"


class TestCallableChat:
    def test_plain_callable(self):
        llm = CallableChat(lambda messages: "reply")
        assert llm.complete([{"role": "user", "content": "x"}]) == "reply"

    def test_kwargs_forwarded_when_accepted(self):
        seen = {}

        def fn(messages, max_tokens=None, json_schema=None):
            seen.update(max_tokens=max_tokens, json_schema=json_schema)
            return "ok"

        CallableChat(fn).complete(
            [{"role": "user", "content": "x"}], max_tokens=7, json_schema={"type": "object"}
        )
        assert seen == {"max_tokens": 7, "json_schema": {"type": "object"}}

    def test_kwargs_omitted_when_not_accepted(self):
        def fn(messages):
            return "ok"

        # must not raise TypeError
        assert CallableChat(fn).complete([{"role": "user", "content": "x"}]) == "ok"

    def test_var_keyword_callable(self):
        def fn(messages, **kwargs):
            return str(kwargs["max_tokens"])

        assert CallableChat(fn).complete([{"role": "user", "content": "x"}], max_tokens=3) == "3"

    def test_name(self):
        def my_gateway(messages):
            return "ok"

        assert "my_gateway" in CallableChat(my_gateway).name
        assert "custom" in CallableChat(lambda m: "ok", name="custom").name


class TestUsageStats:
    def test_total_and_add(self):
        from aspectkit.llm.base import UsageStats

        a = UsageStats(calls=1, prompt_tokens=10, completion_tokens=5)
        b = UsageStats(calls=2, prompt_tokens=3, completion_tokens=7)
        assert a.total_tokens == 15
        total = a + b
        assert (total.calls, total.prompt_tokens, total.completion_tokens) == (3, 13, 12)
        assert total.total_tokens == 25


class TestTransientDetection:
    def test_status_code_and_name(self):
        from aspectkit.llm.base import is_transient_error

        class RateLimitError(Exception):
            status_code = 429

        class BadRequestError(Exception):
            status_code = 400

        assert is_transient_error(RateLimitError())
        assert not is_transient_error(BadRequestError())
        assert is_transient_error(type("OverloadedError", (Exception,), {})())
        assert not is_transient_error(ValueError("nope"))
