import base64
import io
import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

_server_started = False


def start_server(port=8000):
    """Start a background HTTP server to serve log.html."""
    global _server_started
    if _server_started:
        return
    _server_started = True

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.path = "/log.html"
            return super().do_GET()
        def log_message(self, *args):
            pass

    server = HTTPServer(("", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"  log: http://localhost:{port}")


# ARC-AGI official color palette
PALETTE = [
    (0xFF, 0xFF, 0xFF),  # 0  white
    (0xCC, 0xCC, 0xCC),  # 1  off-white
    (0x99, 0x99, 0x99),  # 2  light-gray
    (0x66, 0x66, 0x66),  # 3  gray
    (0x33, 0x33, 0x33),  # 4  off-black
    (0x00, 0x00, 0x00),  # 5  black
    (0xE5, 0x3A, 0xA3),  # 6  magenta
    (0xFF, 0x7B, 0xCC),  # 7  light-magenta
    (0xF9, 0x3C, 0x31),  # 8  red
    (0x1E, 0x93, 0xFF),  # 9  blue
    (0x88, 0xD8, 0xF1),  # 10 light-blue
    (0xFF, 0xDC, 0x00),  # 11 yellow
    (0xFF, 0x85, 0x1B),  # 12 orange
    (0x92, 0x12, 0x31),  # 13 maroon
    (0x4F, 0xCC, 0x30),  # 14 green
    (0xA3, 0x56, 0xD6),  # 15 purple
]


def _render_grid(grid_layers):
    """Render a 64x64 grid to a small PNG. Returns base64 string or None."""
    if not grid_layers:
        return None
    try:
        import numpy as np
        from PIL import Image

        grid = np.array(grid_layers[-1], dtype=np.uint8)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        for val, color in enumerate(PALETTE):
            mask = grid == val
            img[mask] = color

        image = Image.fromarray(img).resize((256, 256), Image.NEAREST)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "input_200k": 0.50, "output": 3.00, "output_200k": 3.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "input_200k": 4.00, "output": 12.00, "output_200k": 18.00},
}


def write_log(contents, path="log.html", usage=None, model=None, **_):
    """Render contents list to HTML — faithful to what Gemini sees."""
    usage = usage or {}
    parts_html = []

    for content in contents:
        role = getattr(content, "role", "?")

        for part in content.parts:
            if part.text:
                parts_html.append(
                    f'<div class="msg {role}"><b>{role} / text</b>'
                    f'<pre>{_esc(part.text)}</pre></div>'
                )

            elif part.function_call:
                fc = part.function_call
                parts_html.append(
                    f'<div class="msg call"><b>model / function_call: {_esc(fc.name)} (id={fc.id})</b>'
                    f'<pre>{_esc(json.dumps(dict(fc.args), indent=2))}</pre></div>'
                )

            elif part.function_response:
                fr = part.function_response
                result = fr.response
                result_str = json.dumps(result, indent=2, default=str)
                if len(result_str) > 5000:
                    result_str = result_str[:5000] + "\n... (truncated)"

                block = f'<div class="msg response"><b>user / function_response: {_esc(fr.name)}</b>'
                block += f'<pre>{_esc(result_str)}</pre>'

                # render inline image if function response has multimodal parts
                if fr.parts:
                    for frp in fr.parts:
                        if frp.inline_data and frp.inline_data.data:
                            mime = frp.inline_data.mime_type or "image/png"
                            b64 = base64.b64encode(frp.inline_data.data).decode()
                            block += f'<img src="data:{mime};base64,{b64}" style="max-width:256px;image-rendering:pixelated">'

                block += '</div>'
                parts_html.append(block)

            elif part.inline_data:
                mime = part.inline_data.mime_type or "application/octet-stream"
                size = len(part.inline_data.data)
                if mime.startswith("image/"):
                    b64 = base64.b64encode(part.inline_data.data).decode()
                    parts_html.append(
                        f'<div class="msg inline"><b>user / inline_data ({mime}, {size} bytes)</b>'
                        f'<img src="data:{mime};base64,{b64}" style="max-width:512px"></div>'
                    )
                else:
                    parts_html.append(
                        f'<div class="msg inline"><b>user / inline_data ({mime}, {size} bytes)</b>'
                        f'<pre>[binary data]</pre></div>'
                    )

    # build usage bar
    prompt_t = usage.get("prompt_tokens", 0)          # current context size
    prompt_total = usage.get("prompt_tokens_total", 0) # cumulative input across all calls
    output_t = usage.get("output_tokens", 0)           # cumulative output
    thinking_t = usage.get("thinking_tokens", 0)       # cumulative thinking
    prices = PRICING.get(model or "", PRICING["gemini-3-flash-preview"])
    in_price = prices["input_200k"] if prompt_t > 200_000 else prices["input"]
    out_price = prices["output_200k"] if prompt_t > 200_000 else prices["output"]
    cost_in = prompt_total * in_price / 1_000_000
    cost_out = (output_t + thinking_t) * out_price / 1_000_000
    cost = cost_in + cost_out
    usage_html = (
        f'<div class="usage">'
        f'context: {prompt_t:,} / 1,048,576 &nbsp; '
        f'output: {output_t:,} &nbsp; '
        f'thinking: {thinking_t:,} &nbsp; '
        f'~${cost:.4f}'
        f'</div>'
    )

    html = _TEMPLATE.replace("{{USAGE}}", usage_html).replace("{{BODY}}", "\n".join(parts_html))
    Path(path).write_text(html)


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>agent log</title>
<style>
  body { font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; max-width: 1000px; margin: 0 auto; }
  .usage { position: fixed; top: 0; right: 0; padding: 8px 16px; background: #222; border-bottom-left-radius: 8px; color: #aaa; font-size: 11px; z-index: 100; }
  .msg { margin: 6px 0; padding: 8px; border-radius: 4px; }
  .user { background: #1e3a5f; }
  .model { background: #2a2a2a; }
  .call { background: #3a2a1a; border-left: 3px solid #ff9800; }
  .response { background: #1a2a1a; border-left: 3px solid #4caf50; }
  .inline { background: #2a2a3a; }
  pre { white-space: pre-wrap; word-wrap: break-word; margin: 4px 0; font-size: 12px; }
  b { display: block; margin-bottom: 4px; color: #888; font-size: 11px; }
  img { border-radius: 4px; margin: 4px 0; }
</style>
</head>
<body>
{{USAGE}}
{{BODY}}
<script>window.scrollTo(0, document.body.scrollHeight);</script>
</body>
</html>
"""
