from typing import Any, Optional

import docker
import io
import os
import tarfile


MAX_OUTPUT = 30_000
DEFAULT_TIMEOUT = 60_000
MAX_TIMEOUT = 600_000
DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000


class SandboxOrchestrator:
    """Tool call harness in a container"""

    def __init__(self, name: str = "sandbox", workdir: str = "/home/agent"):
        self.name = name
        self.workdir = workdir
        self._client = docker.from_env()
        self._client.ping()
        self._image = self._client.images.get("jackgamer-sandbox")
        self._c = self._create_container()
        self.func_callable_map = {
            "bash": self.bash,
            "view": self.view,
            "write": self.write,
            "edit": self.edit,
        }

    def _create_container(self):
        """Create a fresh container from the image."""
        # remove old container if it exists
        try:
            old = self._client.containers.get(self.name)
            old.remove(force=True)
        except docker.errors.NotFound:
            pass
        c = self._client.containers.run(
            self._image,
            name=self.name,
            detach=True,
            mem_limit="4g",
            nano_cpus=2_000_000_000,  # 2 CPUs
            security_opt=["no-new-privileges:true"],
        )
        return c

    def reset(self):
        """Nuke container, start fresh from image."""
        self._c = self._create_container()

    def _resolve(self, path: str) -> str:
        if not os.path.isabs(path):
            return os.path.join(self.workdir, path)
        return path

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
            numbered.append(f"{i + offset + 1}\t{line}")
        result = "\n".join(numbered)
        if offset + limit < len(all_lines):
            result += f"\n\n(File has more lines. Use 'offset' parameter to read beyond line {offset + len(selected)})"
        return result

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        raw = self.read_file(file_path).decode("utf-8", errors="replace")
        count = raw.count(old_string)
        if count == 0:
            return f"error: old_string not found in {file_path}"
        if count > 1 and not replace_all:
            return f"error: old_string has {count} matches. Use replace_all=true or provide more context."
        if replace_all:
            new = raw.replace(old_string, new_string)
        else:
            new = raw.replace(old_string, new_string, 1)
        self.write_file(file_path, new.encode("utf-8"))
        replaced = count if replace_all else 1
        return f"Replaced {replaced} occurrence(s) in {file_path}"

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

    def execute_tool(self, name: str, args: Optional[dict[str, Any]] = None):
        try:
            return {"result": self.func_callable_map[name](**(args or {}))}
        except Exception as e:
            return {"result": f"error: {type(e).__name__}: {e}"}
