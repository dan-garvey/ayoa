"""Tests for SSE streaming format."""


class TestSSEFormat:
    def test_sse_event_format(self):
        from app.api.story_routes import _sse_event
        result = _sse_event("status", {"phase": "started"})
        assert result.startswith("event: status\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_sse_event_json(self):
        import json
        from app.api.story_routes import _sse_event
        result = _sse_event("chunk", {"text": "hello"})
        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        data = json.loads(data_line[6:])
        assert data["text"] == "hello"
