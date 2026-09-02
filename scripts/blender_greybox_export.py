#!/usr/bin/env python3
"""把 Blender 場景輸出成「灰模影片」給影片模型當運鏡參考，或把 playground 的灰模場景檔
在 Blender 裡重建。

社群裡「用 Blender 做運鏡參考」的實際做法就是：本機 Blender 搭一個沒有材質的粗塊場景、
K 一條相機動畫、算一支低畫質 mp4，上傳當參考影片（提示詞裡用「Video 1」指涉）。這支腳本
把「算出符合規格的那支 mp4」這一步固定下來，省得每次手動調 Workbench／解析度／編碼設定。

用法（在你自己的 .blend 上，只改算圖設定、不動場景）：

    blender -b my_scene.blend -P scripts/blender_greybox_export.py -- --out greybox.mp4
    blender -b my_scene.blend -P scripts/blender_greybox_export.py -- --out greybox.mp4 --ratio 9:16

用法（從 playground 灰模節點「匯出場景檔」下載的 greybox-scene.txt 重建場景並輸出）：

    blender -b -P scripts/blender_greybox_export.py -- --spec greybox-scene.txt --out greybox.mp4
    blender    -P scripts/blender_greybox_export.py -- --spec greybox-scene.txt   # 不加 -b：建好場景留在 Blender 裡繼續細修

輸出規格（對齊 Canvas 灰模節點與 w3.0 參考影片的限制）：
  - Workbench 引擎、單一中性灰、Studio 光、無外框線；背景淺灰
  - 24 fps、H.264 mp4；解析度照 --ratio（16:9 → 832×480，同節點）
  - 片長上限 --max-seconds（預設 15，w3.0 單支參考影片上限）——超過就截斷並警告

場景檔語法（與 static/js/canvas.js 的 parseGreyboxSpec() 同一套，改一邊要改另一邊）：
  box W H D | cyl R H | cone R H | sphere R | figure H   at x y z [rot deg] [move to x y z]
  camera from x y z to x y z [look x y z]
  # move: dolly_in   # distance: 14   # height: 3   # duration: 3   # ratio: 16:9   ← 檔頭設定註解
座標系：Y 向上、地面 Y=0、相機看向 -Z（three.js 慣例）。轉 Blender（Z 向上、相機看向 +Y）：
(x, y, z) → (x, -z, y)。figure 的 y 是腳底，其他量體的 y 是中心。

⚠️ 未在真的 Blender 裡跑過（撰寫環境沒有 Blender，2026-09-03）。純函式部分
（解析、座標轉換、運鏡公式）有 tests/test_pure_functions.py 鎖住、與 JS 版對照過；
bpy 那段是照 Blender 4.x Python API 寫的，第一次跑若有 API 名稱對不上請回報。
"""
import argparse
import math
import os
import re
import sys

try:
    import bpy  # type: ignore
except ImportError:  # 測試環境沒有 Blender，純函式仍可 import
    bpy = None

FPS = 24
RATIOS = {  # 與 canvas.js 的 GREYBOX_RATIOS 相同
    "16:9": (832, 480), "9:16": (480, 832), "1:1": (640, 640),
    "4:3": (800, 600), "3:4": (600, 800),
}
DIMS = {"box": 3, "cyl": 2, "cone": 2, "sphere": 1, "figure": 1}
GREY = (0.69, 0.69, 0.69)
GROUND = (0.77, 0.77, 0.77)
BACKGROUND = (0.85, 0.85, 0.85)


# ── 純函式（不碰 bpy）────────────────────────────────────────────────────────

def parse_spec(text):
    """回 (shapes, camera, settings, errors)。shapes 每筆：
    {'kind','dims','pos','rot','to'}；camera：{'from','to','look'} 或 None；
    settings：檔頭 `# key: value` 註解（move/distance/height/duration/ratio）。"""
    shapes, errors, settings, camera = [], [], {}, None

    def vec3(tok, i):
        try:
            v = [float(x) for x in tok[i:i + 3]]
        except ValueError:
            return None
        return v if len(v) == 3 else None

    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.match(r"#\s*(move|distance|height|duration|ratio)\s*:\s*(\S+)", line)
            if m:
                settings[m.group(1)] = m.group(2)
            continue
        tok = line.split()
        kind = tok[0].lower()
        if kind == "camera":
            fi = tok.index("from") if "from" in tok else -1
            ti = tok.index("to") if "to" in tok else -1
            li = tok.index("look") if "look" in tok else -1
            frm = vec3(tok, fi + 1) if fi >= 0 else None
            to = vec3(tok, ti + 1) if ti >= 0 else None
            if not frm or not to:
                errors.append(f"第 {ln} 行：camera 要寫成「camera from x y z to x y z [look x y z]」")
                continue
            look = vec3(tok, li + 1) if li >= 0 else None
            if li >= 0 and not look:
                errors.append(f"第 {ln} 行：look 後面要接三個數字")
                continue
            camera = {"from": frm, "to": to, "look": look}
            continue
        at = tok.index("at") if "at" in tok else -1
        ri = tok.index("rot") if "rot" in tok else -1
        mi = tok.index("move") if "move" in tok else -1
        pos = vec3(tok, at + 1) if at >= 0 else [0.0, 0.0, 0.0]
        if not pos:
            errors.append(f"第 {ln} 行：at 後面要接三個數字（x y z）")
            continue
        rot = 0.0
        if ri >= 0:
            try:
                rot = float(tok[ri + 1])
            except (ValueError, IndexError):
                rot = 0.0
        to = None
        if mi >= 0:
            to = vec3(tok, mi + 2) if len(tok) > mi + 1 and tok[mi + 1] == "to" else None
            if not to:
                errors.append(f"第 {ln} 行：move 要寫成「move to x y z」")
                continue
        end = min(i for i in (at, ri, mi, len(tok)) if i >= 0)
        need = DIMS.get(kind)
        try:
            dims = [float(x) for x in tok[1:end]]
        except ValueError:
            dims = []
        if need is None or len(dims) < need:
            errors.append(f"第 {ln} 行：看不懂「{line}」")
            continue
        shapes.append({"kind": kind, "dims": dims[:need], "pos": pos, "rot": rot, "to": to})
    return shapes, camera, settings, errors


def to_blender(v):
    """three.js（Y 上、看 -Z）→ Blender（Z 上、看 +Y）。行列式 +1，旋轉方向不變。"""
    x, y, z = v
    return (x, -z, y)


def smoothstep(t):
    return t * t * (3 - 2 * t)


def lerp(a, b, k):
    return [a[i] + (b[i] - a[i]) * k for i in range(3)]


def camera_move(name, t, d, h):
    """內建七種運鏡，與 canvas.js 的 GREYBOX_MOVES 逐字對應。回 (pos, look)，three.js 座標。"""
    if name == "dolly_in":
        return [0, h, d - (d - d * 0.35) * t], [0, h * 0.6, -d]
    if name == "dolly_out":
        return [0, h, d * 0.35 + (d - d * 0.35) * t], [0, h * 0.6, -d]
    if name in ("orbit_left", "orbit_right"):
        a = (-1 if name == "orbit_left" else 1) * math.pi / 4 * t
        return [math.sin(a) * d, h, math.cos(a) * d], [0, h * 0.6, 0]
    if name == "crane_up":
        return [0, h * 0.4 + h * 1.6 * t, d], [0, h * 0.5, -d * 0.5]
    if name == "push_through":
        return [0, h, d - d * 1.8 * t], [0, h, -d]
    if name == "pan_right":
        return [0, h, d], [-d * 0.6 + d * 1.2 * t, h * 0.6, -d]
    raise ValueError(f"未知的運鏡：{name}")


def scene_centre(shapes):
    """camera 行沒寫 look 時的注視點：所有量體 at 位置的包圍盒中心（粗略，夠用）。"""
    if not shapes:
        return [0.0, 1.0, -6.0]
    pts = [s["pos"] for s in shapes]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return [(lo[i] + hi[i]) / 2 for i in range(3)]


def frame_count(duration, fps=FPS):
    return max(2, int(round(duration * fps)))


# ── bpy 部分 ────────────────────────────────────────────────────────────────

def _grey_material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1.0)
    return mat


def _add_mesh(kind, dims, mat):
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.active_object
        obj.scale = (dims[0], dims[2], dims[1])   # 我們的 (W,H,D) → Blender (X,Y,Z)=(W,D,H)
    elif kind == "cyl":
        bpy.ops.mesh.primitive_cylinder_add(radius=dims[0], depth=dims[1], vertices=24)
        obj = bpy.context.active_object
    elif kind == "cone":
        bpy.ops.mesh.primitive_cone_add(radius1=dims[0], radius2=0, depth=dims[1], vertices=24)
        obj = bpy.context.active_object
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=dims[0], segments=24, ring_count=16)
        obj = bpy.context.active_object
    obj.data.materials.append(mat)
    return obj


def build_figure(h, mat):
    """人形假人，比例與 canvas.js 的 _buildFigure() 相同；回傳原點在腳底的父物件。"""
    bpy.ops.object.empty_add(type="PLAIN_AXES")
    root = bpy.context.active_object
    root.name = "Figure"
    parts = [
        ("cyl", [h * 0.09, h * 0.47], (0, 0, h * 0.235)),
        ("box", [h * 0.24, h * 0.33, h * 0.13], (0, 0, h * 0.635)),
        ("cyl", [h * 0.035, h * 0.3], (-h * 0.16, 0, h * 0.63)),
        ("cyl", [h * 0.035, h * 0.3], (h * 0.16, 0, h * 0.63)),
        ("sphere", [h * 0.08], (0, 0, h * 0.9)),
    ]
    for kind, dims, loc in parts:
        o = _add_mesh(kind, dims, mat)
        o.location = loc
        o.parent = root
    return root


def _keyframe_location(obj, frames_positions, interpolation):
    for f, p in frames_positions:
        obj.location = p
        obj.keyframe_insert(data_path="location", frame=f)
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = interpolation


def build_scene(shapes, camera, settings, duration):
    scene = bpy.context.scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    n = frame_count(duration)
    scene.frame_start, scene.frame_end = 1, n

    grey = _grey_material("GreyboxGrey", GREY)
    ground_mat = _grey_material("GreyboxGround", GROUND)
    bpy.ops.mesh.primitive_plane_add(size=400)
    bpy.context.active_object.data.materials.append(ground_mat)

    for s in shapes:
        obj = build_figure(s["dims"][0], grey) if s["kind"] == "figure" else _add_mesh(s["kind"], s["dims"], grey)
        obj.location = to_blender(s["pos"])
        obj.rotation_euler = (0, 0, math.radians(s["rot"]))
        if s["to"]:
            # 與節點一致：緩入緩出。Blender 的 BEZIER 兩個關鍵格就是這個形狀
            _keyframe_location(obj, [(1, to_blender(s["pos"])), (n, to_blender(s["to"]))], "BEZIER")

    # 相機 + 注視點 Empty（Track To 約束），跟節點的 lookAt() 等價
    bpy.ops.object.empty_add(type="PLAIN_AXES")
    look = bpy.context.active_object
    look.name = "GreyboxLook"
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "GreyboxCamera"
    cam.data.sensor_fit = "VERTICAL"    # 節點的 PerspectiveCamera(40°) 是垂直視角
    cam.data.angle_y = math.radians(40)
    con = cam.constraints.new(type="TRACK_TO")
    con.target = look
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    scene.camera = cam

    if camera:
        target = camera["look"] or scene_centre(shapes)
        _keyframe_location(cam, [(1, to_blender(camera["from"])), (n, to_blender(camera["to"]))], "BEZIER")
        look.location = to_blender(target)
    else:
        move = settings.get("move", "dolly_in")
        d = float(settings.get("distance", 14))
        h = float(settings.get("height", 3))
        cam_frames, look_frames = [], []
        for i in range(n):
            t = i / (n - 1)
            pos, lk = camera_move(move, t, d, h)
            cam_frames.append((i + 1, to_blender(pos)))
            look_frames.append((i + 1, to_blender(lk)))
        _keyframe_location(cam, cam_frames, "LINEAR")
        _keyframe_location(look, look_frames, "LINEAR")
    return scene


def apply_render_settings(scene, ratio, out_path, max_seconds):
    w, h = RATIOS[ratio]
    r = scene.render
    r.engine = "BLENDER_WORKBENCH"
    r.resolution_x, r.resolution_y, r.resolution_percentage = w, h, 100
    r.fps, r.fps_base = FPS, 1.0
    r.film_transparent = False
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = GREY
    sh.show_object_outline = False
    sh.show_shadows = False
    sh.show_cavity = False
    sh.show_specular_highlight = False
    scene.display.render_aa = "5"
    world = scene.world or bpy.data.worlds.new("GreyboxWorld")
    scene.world = world
    world.color = BACKGROUND
    r.image_settings.file_format = "FFMPEG"
    r.ffmpeg.format = "MPEG4"
    r.ffmpeg.codec = "H264"
    r.ffmpeg.constant_rate_factor = "MEDIUM"
    r.ffmpeg.gopsize = FPS
    r.ffmpeg.audio_codec = "NONE"
    r.filepath = os.path.abspath(out_path)
    r.use_file_extension = True

    max_frames = int(max_seconds * FPS)
    length = scene.frame_end - scene.frame_start + 1
    if length > max_frames:
        print(f"[greybox] 場景 {length / FPS:.1f} 秒超過上限 {max_seconds} 秒，只輸出前 {max_seconds} 秒", file=sys.stderr)
        scene.frame_end = scene.frame_start + max_frames - 1
    if scene.camera is None:
        raise SystemExit("[greybox] 場景沒有作用中的相機（scene.camera），無法輸出")


def _find_output(out_path):
    """Blender 對影片輸出有時會在檔名後面補上影格範圍（例如 out0001-0072.mp4）；找出實際檔案。"""
    out_path = os.path.abspath(out_path)
    if os.path.exists(out_path):
        return out_path
    base, ext = os.path.splitext(out_path)
    d = os.path.dirname(out_path) or "."
    for name in sorted(os.listdir(d)):
        full = os.path.join(d, name)
        if full.startswith(base) and name.endswith(ext or ".mp4") and full != out_path:
            return full
    return None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", help="playground 灰模節點匯出的場景檔（不給就用目前開啟的 .blend）")
    ap.add_argument("--out", help="輸出 mp4 路徑；不給就只建場景不算圖")
    ap.add_argument("--ratio", choices=sorted(RATIOS), help="畫面比例（預設：場景檔的 # ratio，或 16:9）")
    ap.add_argument("--duration", type=float, help="片長秒數（只在 --spec 模式有意義；預設：場景檔的 # duration，或 3）")
    ap.add_argument("--max-seconds", type=float, default=15, help="片長上限，超過截斷（預設 15）")
    args = ap.parse_args(argv)

    if bpy is None:
        raise SystemExit("這支腳本要在 Blender 裡跑：blender -b -P scripts/blender_greybox_export.py -- ...")

    settings = {}
    if args.spec:
        with open(args.spec, encoding="utf-8") as f:
            shapes, camera, settings, errors = parse_spec(f.read())
        for e in errors:
            print("[greybox] " + e, file=sys.stderr)
        if not shapes:
            raise SystemExit("[greybox] 場景檔裡沒有任何量體")
        duration = args.duration or float(settings.get("duration", 3))
        scene = build_scene(shapes, camera, settings, duration)
        print(f"[greybox] 已建立 {len(shapes)} 個量體，{duration} 秒" + ("，自訂相機路徑" if camera else f"，運鏡 {settings.get('move', 'dolly_in')}"))
    else:
        scene = bpy.context.scene

    if not args.out:
        print("[greybox] 沒有 --out，只建場景不算圖")
        return
    ratio = args.ratio or settings.get("ratio", "16:9")
    if ratio not in RATIOS:
        raise SystemExit(f"[greybox] 不支援的比例 {ratio}，可用：{', '.join(sorted(RATIOS))}")
    apply_render_settings(scene, ratio, args.out, args.max_seconds)
    bpy.ops.render.render(animation=True)
    actual = _find_output(args.out)
    if actual and actual != os.path.abspath(args.out):
        os.replace(actual, os.path.abspath(args.out))
        actual = os.path.abspath(args.out)
    w, h = RATIOS[ratio]
    secs = (scene.frame_end - scene.frame_start + 1) / FPS
    print(f"[greybox] 完成：{actual}（{w}×{h}，{secs:.1f} 秒，{FPS} fps）")


if __name__ == "__main__":
    # Blender 把自己的參數放前面，腳本參數在 "--" 之後
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)
