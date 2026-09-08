"""守門：新增的檔案有沒有被複製進 Docker image。

起因（2026-09-09）：admin.py 加進 app.py 的 import，但 Dockerfile 只 COPY app.py／
templates/／static/，部署後容器一啟動就 `ModuleNotFoundError: No module named 'admin'`。
本機測試全過、CI 也只跑測試，所以直到 Cloud Run 健康檢查失敗才發現。
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")


def _copied_paths() -> set[str]:
    """Dockerfile 裡 COPY 進來的來源路徑（含結尾斜線的目錄）。"""
    out = set()
    for line in DOCKERFILE.splitlines():
        m = re.match(r"\s*COPY\s+(?!--)(\S+)\s+\S+", line)
        if m:
            out.add(m.group(1))
    return out


def _local_modules_imported_by(path: Path) -> set[str]:
    """該檔 import 的「本專案頂層模組」（repo 根目錄有同名 .py 的才算）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return {n for n in names if (ROOT / f"{n}.py").exists() and n != path.stem}


def test_imported_local_modules_are_copied_into_image():
    copied = _copied_paths()
    missing = []
    seen: set[str] = set()
    queue = ["app"]
    while queue:
        mod = queue.pop()
        if mod in seen:
            continue
        seen.add(mod)
        for dep in _local_modules_imported_by(ROOT / f"{mod}.py"):
            queue.append(dep)
            if f"{dep}.py" not in copied:
                missing.append(dep)
    assert not missing, f"這些模組被 import 但沒 COPY 進 image：{sorted(set(missing))}"


def test_scripts_dir_copied_because_admin_loads_usage_stats():
    """admin.py 用 importlib 從 scripts/usage_stats.py 載入報表函式——那是執行期依賴，
    靜態 import 掃不到，所以單獨釘住。"""
    admin_src = (ROOT / "admin.py").read_text(encoding="utf-8")
    assert "scripts" in admin_src and "usage_stats.py" in admin_src
    assert "scripts/" in _copied_paths(), "admin.py 讀 scripts/usage_stats.py，Dockerfile 要 COPY scripts/"
