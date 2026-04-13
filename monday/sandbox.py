from typing import Any, Optional

import docker
import io
import os
import tarfile
import subprocess

MAX_OUTPUT = 30_000
DEFAULT_TIMEOUT = 60_000
MAX_TIMEOUT = 600_000
DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000


class SandboxOrchestrator:
    """Tool call harness in a container"""

    def __init__(self, name: str = "sandbox", workdir: str = "/home/agent"):
        self.name = name  # FKEY: scorecard id
        self.workdir = workdir
        self._client = docker.from_env()
        self._client.ping()
        self._c = self._client.containers.get(name)
        self._c.reload()
        self.reset()
        self.func_callable_map = {
            "bash": self.bash,
            "view": self.view,
            "write": self.write,
        }

    def reset(self):
        self._c.restart(timeout=5)
        self._c.reload()

    # TOOLS
    def quote(self, s: str) -> str:
        return "'" + s.replace("'", "'\\''") + "'"

    def truncate(self, text: str) -> str:
        if len(text) <= MAX_OUTPUT:
            return text
        half = MAX_OUTPUT // 2
        middle = text[half: len(text) - half]
        truncated_lines = middle.count("\n") + 1
        return f"{text[:half]}\n\n... [{truncated_lines} lines truncated] ...\n\n{text[-half:]}"

    def bash(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        timeout = max(1, min(timeout, MAX_TIMEOUT))
        timeout_s = timeout // 1000
        code, out = self._c.exec_run(
            ["bash", "-c",
                f"timeout {timeout_s} bash -c {self.quote(command)}"],
            workdir=self.workdir,
            demux=False,
        )
        text = self.truncate(out.decode("utf-8", errors="replace").strip())
        if code != 0:
            text = f"{text}\nExit code {code}".strip()
        return text or "no output"

    def view(
        self, file_path: str, offset: int = 0, limit: int = DEFAULT_READ_LIMIT
    ) -> str:
        raw = self.read_file(file_path)
        all_lines = raw.decode("utf-8", errors="replace").splitlines()
        selected = all_lines[offset: offset + limit]
        numbered = []
        for i, line in enumerate(selected):
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "..."
            numbered.append(f"{i + offset + 1:6}|{line}")
        result = "<file>\n" + "\n".join(numbered) + "\n</file>"
        if offset + limit < len(all_lines):
            result += f"\n\n(File has more lines. Use 'offset' parameter to read beyond line {offset + len(selected)})"
        return result

    def write(self, file_path: str, content: str) -> str:
        self.bash(f"mkdir -p {self.quote(os.path.dirname(file_path))}")
        self.write_file(file_path, content.encode("utf-8"))
        return f"File successfully written: {file_path}"

    def read_file(self, path: str) -> bytes:
        bits, _ = self._c.get_archive(path)
        tar_buf = io.BytesIO(b"".join(bits))
        with tarfile.open(fileobj=tar_buf) as tar:
            return tar.extractfile(tar.getmembers()[0]).read()

    def write_file(self, path: str, content: bytes):
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            info = tarfile.TarInfo(name=os.path.basename(path))
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        tar_buf.seek(0)
        self._c.put_archive(os.path.dirname(path), tar_buf.getvalue())

    def execute_tool(self, name: str, args: Optional[dict[str, Any]]):
        res = self.func_callable_map[name](**args)

    def execute_tool_with_timeout(self, name: str, args: Optional[dict[str, Any]]):
        try:
            return self.execute_tool(name, args)
        except subprocess.TimeoutExpired:
            return {"result": "error: command timed out"}
        except Exception as e:
            return {"result": f"error: {type(e).__name__}: {e}"}
