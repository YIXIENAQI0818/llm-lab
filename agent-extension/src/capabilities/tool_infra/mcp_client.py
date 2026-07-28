"""MCP 客户端 — 通过子进程 + JSON-RPC 连接远程 MCP Server。

纯 Python 标准库实现（subprocess + json + threading），不依赖第三方 MCP 包。

协议: MCP (Model Context Protocol) over stdio
- 客户端启动 Server 子进程，通过 stdin/stdout 交换 JSON-RPC 2.0 消息
- 每行一条完整的 JSON-RPC 消息，以 \n 分隔
- 请求带 id，响应回传相同 id；通知不带 id，无需响应

启动流程:
  1. subprocess.Popen 启动 Server
  2. initialize 请求握手（协商协议版本和能力）
  3. initialized 通知（告知 Server 客户端已就绪）
  4. tools/list 请求发现工具（批量注册到 ToolRegistry）
  5. 对话中收到工具调用 → tools/call 请求 → 返回结果
"""

import json
import subprocess
import threading
import uuid


# MCP 服务器配置列表。
# 每个配置:
#   command: 启动命令（list，如 ["npx", "-y", "@anthropic/mcp-server-github"]）
#   namespace: 工具名前缀（如 "github__"，避免不同 Server 工具名冲突）
#   env: 额外的环境变量（dict，可选）
#
# 示例:
# MCP_SERVERS = [
#     {
#         "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
#         "namespace": "fs__",
#     },
#     {
#         "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
#         "namespace": "github__",
#         "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"},
#     },
# ]
MCP_SERVERS: list[dict] = [
    {
        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem",
                    "/home/yixienaqi0818/workspace"],
        "namespace": "fs__",
    },
    {
        "command": ["python", "-m", "mcp_server_fetch"],
        "namespace": "fetch__",
    },
    {
        "command": ["python", "-m", "mcp_server_git"],
        "namespace": "git__",
    },
]


class MCPClient:
    """MCP Server 客户端，管理与一个 Server 子进程的通信。

    用法:
        client = MCPClient(command=["npx", "some-mcp-server"], namespace="s__")
        tools = client.discover()        # → [{name, description, inputSchema}, ...]
        result = client.call("s__tool", {"arg": "val"})  # → "result string"
        client.close()
    """

    def __init__(self, command: list[str], namespace: str = "",
                 env: dict | None = None):
        self.namespace = namespace
        self._tools: list[dict] = []
        self._lock = threading.Lock()
        self._id_counter = 0

        # 启动子进程
        import os
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=proc_env,
        )

        # 握手: initialize → 收到响应 → initialized 通知
        self._initialize()

    # ================================================================
    # 工具发现
    # ================================================================

    def discover(self) -> list[dict]:
        """发现 Server 提供的工具列表。

        Returns:
            [{name, description, inputSchema}, ...]
            其中 inputSchema 是 JSON Schema 格式的参数定义。
        """
        if self._tools:
            return self._tools

        result = self._send_request("tools/list", {})
        tools = result.get("tools", [])

        # 规范化：确保每个工具都有 description 和 inputSchema
        for t in tools:
            t.setdefault("description", "")
            t.setdefault("inputSchema", {"type": "object", "properties": {}})

        self._tools = tools
        return self._tools

    # ================================================================
    # 工具调用
    # ================================================================

    def call(self, tool_name: str, arguments: dict) -> str:
        """调用远程工具，返回结果字符串。"""
        result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        # MCP 返回格式: {content: [{type: "text", text: "..."}]}
        content = result.get("content", [])
        if isinstance(content, list) and content:
            # 拼接所有 text 类型的内容
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, dict):
                    parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(parts) if parts else json.dumps(
                result, ensure_ascii=False,
            )

        # 非标准格式：直接序列化整个 result
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    def _make_caller(self, tool_name: str):
        """返回闭包 fn(**kwargs) → self.call(tool_name, kwargs)。

        ToolRegistry 用此方法将 MCP 工具包装为与本地工具一致的 callable。
        """
        def caller(**kwargs):
            return self.call(tool_name, kwargs)
        return caller

    # ================================================================
    # 生命周期
    # ================================================================

    def close(self):
        """终止子进程。"""
        try:
            self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        except Exception:
            pass

    # ================================================================
    # JSON-RPC 通信
    # ================================================================

    def _initialize(self):
        """MCP 握手: initialize 请求 + initialized 通知。"""
        result = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "agent-extension",
                "version": "0.1.0",
            },
        })
        # 检查协商结果
        server_version = result.get("protocolVersion", "unknown")
        server_name = result.get("serverInfo", {}).get("name", "unknown")
        print(f"[MCP] {server_name} (协议 {server_version}) 已连接")

        # 发送 initialized 通知（无需响应）
        self._send_notification("notifications/initialized", {})

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _send_request(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        req_id = self._next_id()
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        return self._send_and_wait(request, req_id)

    def _send_notification(self, method: str, params: dict):
        """发送 JSON-RPC 通知（无 id，不等待响应）。"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        with self._lock:
            self._write_line(notification)
        # 通知无需响应，直接返回

    def _send_and_wait(self, request: dict, req_id: int) -> dict:
        """发送请求并阻塞等待匹配 id 的响应。"""
        with self._lock:
            self._write_line(request)

            # 逐行读取 stdout，找到匹配 id 的响应
            while True:
                line = self._process.stdout.readline()
                if not line:
                    raise MCPError(
                        f"MCP Server 已断开（发送 {request.get('method')} 后）"
                    )
                line = line.strip()
                if not line:
                    continue

                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    # 非 JSON 输出（某些 Server 会在 stdout 打日志），跳过
                    continue

                if response.get("id") == req_id:
                    # 检查错误
                    if "error" in response:
                        err = response["error"]
                        raise MCPError(
                            f"MCP 错误 [{err.get('code')}]: {err.get('message')}"
                        )
                    return response.get("result", {})

                # 非匹配 id 的响应（如并发场景），跳过
                # 注意：当前设计为串行调用，这条路径不应触发

    def _write_line(self, data: dict):
        """写入一行 JSON 到子进程 stdin。"""
        line = json.dumps(data, ensure_ascii=False)
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
        except BrokenPipeError:
            raise MCPError("MCP Server 连接已断开（BrokenPipe）")


class MCPError(Exception):
    """MCP 通信错误。"""
    pass


# ================================================================
# 模块级工厂
# ================================================================

def get_mcp_clients() -> list[MCPClient]:
    """根据 MCP_SERVERS 配置创建 MCPClient 列表。

    在 Agent 初始化时由 ToolRegistry 调用。
    如果某个 Server 启动失败，打印警告并跳过（不影响其他 Server）。
    """
    clients = []
    for cfg in MCP_SERVERS:
        try:
            client = MCPClient(
                command=cfg["command"],
                namespace=cfg.get("namespace", ""),
                env=cfg.get("env"),
            )
            clients.append(client)
        except Exception as e:
            print(f"[MCP] 警告: 跳过 {cfg['command'][0]}: {e}")
    return clients
