"""Persistent Codex app-server transport; retain only public output and summaries."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import defaultdict


def public_item(item):
    kind = item.get("type")
    if kind == "reasoning":
        return {"type": kind, "id": item.get("id"), "summary": item.get("summary", [])}
    if kind == "agentMessage":
        return {k: item[k] for k in ("type", "id", "text", "phase") if k in item}
    if kind == "userMessage":
        return {"type": kind, "id": item.get("id"), "content": item.get("content", [])}
    return {"type": kind, "id": item.get("id")}


def public_result(value):
    if isinstance(value, list):
        return [public_result(v) for v in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "reasoning":
        return public_item(value)
    return {
        k: public_result(v)
        for k, v in value.items()
        if k not in {"encrypted_content", "encryptedContent"}
    }


class AppServer:
    def __init__(self, workspace):
        self.lock = threading.Lock()
        self.waiters = {}
        self.notifications = defaultdict(queue.Queue)
        self.counter = 0
        command = [
            "codex",
            "--strict-config",
            "app-server",
            "--stdio",
            "--config",
            "project_doc_max_bytes=0",
            "--config",
            'web_search="disabled"',
            "--config",
            'model_reasoning_effort="max"',
            "--config",
            'model_reasoning_summary="detailed"',
            "--enable",
            "skip_host_skill_discovery",
        ]
        for feature in (
            "shell_tool",
            "view_image",
            "image_generation",
            "browser_use",
            "browser_use_external",
            "in_app_browser",
            "computer_use",
            "apps",
            "plugins",
            "multi_agent",
            "memories",
            "goals",
            "hooks",
            "skill_search",
            "workspace_dependencies",
        ):
            command.extend(["--disable", feature])
        self.command = command
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=workspace,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.rpc(
            "initialize",
            {
                "clientInfo": {"name": "ayoa_narrative_eval", "version": "1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        self.send({"method": "initialized", "params": {}})

    def _read(self):
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                if "id" in message and "method" not in message:
                    with self.lock:
                        target = self.waiters.get(message["id"])
                    if target:
                        target.put(public_result(message))
                    continue
                if "id" in message:
                    self.send(
                        {
                            "id": message["id"],
                            "error": {
                                "code": -32601,
                                "message": "Interactive tools are unavailable",
                            },
                        }
                    )
                    continue
                method = message.get("method", "")
                params = message.get("params", {})
                thread_id = params.get("threadId")
                if not thread_id:
                    continue
                if method == "item/completed":
                    safe = {
                        "threadId": thread_id,
                        "turnId": params.get("turnId"),
                        "item": public_item(params["item"]),
                    }
                elif method in {
                    "turn/completed",
                    "thread/tokenUsage/updated",
                    "error",
                    "model/rerouted",
                }:
                    safe = public_result(params)
                else:
                    continue
                self.notifications[thread_id].put({"method": method, "params": safe})
        except (ValueError, OSError, KeyError):
            pass
        finally:
            error = {"error": {"message": "App-server connection closed"}}
            with self.lock:
                for target in self.waiters.values():
                    target.put(error)
                for target in self.notifications.values():
                    target.put(error)

    def send(self, message):
        with self.lock:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()

    def rpc(self, method, params, timeout=60):
        target = queue.Queue()
        with self.lock:
            self.counter += 1
            request_id = self.counter
            self.waiters[request_id] = target
        try:
            self.send({"id": request_id, "method": method, "params": params})
            response = target.get(timeout=timeout)
            if "error" in response:
                raise RuntimeError(
                    f"App-server request failed: {method}: {response['error']}"
                )
            return response["result"]
        finally:
            with self.lock:
                self.waiters.pop(request_id, None)

    def turn(self, params, event_sink, timeout=900):
        started = time.monotonic()
        result = self.rpc("turn/start", params)
        turn_id = result["turn"]["id"]
        event_sink({"method": "turn/start/result", "params": public_result(result)})
        thread_id = params["threadId"]
        self.notifications[thread_id]
        items = {}
        usage = None
        while True:
            event = self.notifications[thread_id].get(
                timeout=max(0.01, timeout - (time.monotonic() - started))
            )
            if "error" in event:
                raise RuntimeError("App-server disconnected during generation")
            data = event["params"]
            if data.get("turnId", turn_id) != turn_id:
                continue
            event_sink(event)
            if event["method"] == "item/completed":
                item = data["item"]
                items[item["id"]] = item
            elif event["method"] == "thread/tokenUsage/updated":
                usage = data["tokenUsage"]
            elif event["method"] == "model/rerouted":
                raise RuntimeError("Requested model was rerouted; evidence retained")
            elif event["method"] == "turn/completed":
                turn = data["turn"]
                if turn["id"] != turn_id:
                    continue
                for item in turn.get("items", []):
                    safe = public_item(item)
                    items[safe["id"]] = safe
                final = [
                    i["text"]
                    for i in items.values()
                    if i["type"] == "agentMessage"
                    and i.get("phase") in {None, "final_answer"}
                ]
                summaries = [
                    part
                    for i in items.values()
                    if i["type"] == "reasoning"
                    for part in i.get("summary", [])
                    if isinstance(part, str)
                ]
                return {
                    "executor": "codex-app-server",
                    "status": turn["status"],
                    "exit_code": 0 if turn["status"] == "completed" else 1,
                    "raw": "\n\n".join(final),
                    "reasoning_summaries": summaries,
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "usage": usage,
                    "proxy_elapsed_s": time.monotonic() - started,
                    "item_types": [i["type"] for i in items.values()],
                }

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
