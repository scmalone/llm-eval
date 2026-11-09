#!/usr/bin/env python3
"""Render the vision set (accuracy suite, vision profiles only) and write its tasks.

    python3 tools/make_vision_set.py            # images into <data>/sources/vision-set (a git repo), tasks into
                                                # fixtures/workload/vision/

Ten synthetic images, 1024 x 768, with known answers: six read text (OCR at the sizes that break when image tokens
are capped) and four locate a thing (grounding, graded by box overlap). Synthetic so every answer is exact; the
owner's own screenshots can replace them later with the same task shape.
"""
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from llm_eval.paths import DATA  # noqa: E402
from llm_eval.workload.prepare import TASK_DIR  # noqa: E402

OUT = DATA / "sources" / "vision-set"
W, H = 1024, 768
SANS = "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"
MONO = "/usr/share/fonts/liberation/LiberationMono-Regular.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def canvas(color=(250, 250, 248)):
    image = Image.new("RGB", (W, H), color)
    return image, ImageDraw.Draw(image)


def receipt_grocery():
    image, d = canvas((236, 233, 226))
    d.rectangle([312, 40, 712, 728], fill=(255, 255, 252))
    f, b = font(MONO, 18), font(BOLD, 22)
    d.text((420, 60), "RIVERSIDE MARKET", font=b, fill=(20, 20, 20))
    d.text((352, 92), "412 Canal St  -  Store 0388", font=f, fill=(40, 40, 40))
    lines = [("Organic bananas 1.2kg", "2.39"), ("Sourdough loaf", "5.49"), ("Greek yogurt 750g", "4.99"),
             ("Free-range eggs x12", "6.29"), ("Baby spinach", "3.79"), ("Olive oil 1L", "11.95"),
             ("Cheddar 400g", "7.45"), ("Coffee beans 340g", "2.99")]
    y = 140
    for name, price in lines:
        d.text((340, y), name, font=f, fill=(30, 30, 30))
        d.text((640, y), price, font=f, fill=(30, 30, 30))
        y += 30
    for label, value in (("SUBTOTAL", "45.34"), ("TAX 5.5%", "2.49"), ("TOTAL", "47.83")):
        y += 10 if label == "SUBTOTAL" else 0
        d.text((340, y), label, font=b if label == "TOTAL" else f, fill=(10, 10, 10))
        d.text((628, y), value, font=b if label == "TOTAL" else f, fill=(10, 10, 10))
        y += 32
    d.text((360, y + 30), "THANK YOU - KEEP YOUR RECEIPT", font=f, fill=(60, 60, 60))
    return image


def receipt_hardware():
    image, d = canvas((222, 226, 230))
    d.rectangle([300, 30, 724, 738], fill=(255, 255, 255))
    f, b = font(MONO, 16), font(BOLD, 20)
    d.text((392, 50), "NORTHGATE HARDWARE", font=b, fill=(20, 20, 20))
    items = [("Wood screws 4x40 (x100)", "8.40"), ("M8 hex bolts (x12)", "6.72"), ("M8 nyloc nuts (x12)", "3.18"),
             ("Wall plugs 6mm (x50)", "4.25"), ("Masking tape 24mm", "3.95"), ("Sandpaper P120 (x5)", "5.10"),
             ("Wood glue 250ml", "7.65"), ("Cable ties 200mm (x100)", "6.30"), ("Drill bit HSS 6mm", "4.85"),
             ("Utility blades (x10)", "4.60")]
    y = 100
    for name, price in items:
        d.text((326, y), name, font=f, fill=(30, 30, 30))
        d.text((650, y), price, font=f, fill=(30, 30, 30))
        y += 26
    d.text((326, y + 20), "ITEMS 10        TOTAL 55.00", font=b, fill=(10, 10, 10))
    return image


def terminal():
    image, d = canvas((24, 26, 33))
    f = font(MONO, 15)
    lines = ["$ python3 -m studio.server --port 8732",
             "INFO:     Started server process [41822]",
             "INFO:     Waiting for application startup.",
             "Traceback (most recent call last):",
             '  File "/home/dev/thermal/studio/server.py", line 61, in <module>',
             "    from fastapi_utils.tasks import repeat_every",
             "ModuleNotFoundError: No module named 'fastapi_utils'",
             "ERROR:    Application startup failed. Exiting.",
             "$ "]
    for i, line in enumerate(lines):
        color = (240, 110, 110) if line.startswith(("ModuleNotFound", "ERROR")) else (210, 214, 222)
        d.text((24, 24 + i * 24), line, font=f, fill=color)
    return image


def config_table():
    image, d = canvas()
    f, b = font(SANS, 17), font(BOLD, 17)
    headers = ["Profile", "Context", "KV cache", "CPU MoE", "ubatch"]
    rows = [["harbor-14b", "131,072", "q8_0", "0", "512"], ["kestrel-27b", "65,536", "q8_0", "0", "1024"],
            ["lumen-35b-a3b", "262,144", "q8_0", "9", "512"], ["moraine-30b", "131,072", "f16", "0", "256"],
            ["ostrich-12b", "196,608", "q4_0", "0", "2048"], ["quill-8b", "163,840", "q8_0", "0", "512"]]
    x0, y0, widths = 120, 160, [230, 160, 150, 130, 110]
    d.text((120, 100), "Preset summary (generated)", font=font(BOLD, 24), fill=(20, 20, 20))
    for c, head in enumerate(headers):
        d.text((x0 + sum(widths[:c]) + 8, y0 + 8), head, font=b, fill=(20, 20, 20))
    for r, row in enumerate(rows, 1):
        y = y0 + r * 40
        d.line([x0, y, x0 + sum(widths), y], fill=(200, 200, 200))
        for c, cell in enumerate(row):
            d.text((x0 + sum(widths[:c]) + 8, y + 9), cell, font=f, fill=(40, 40, 40))
    return image


def label_plate():
    image, d = canvas((205, 208, 204))
    d.rectangle([140, 160, 884, 608], fill=(232, 232, 226), outline=(90, 90, 90), width=3)
    d.text((180, 190), "ROOFTOP CONDENSING UNIT", font=font(BOLD, 30), fill=(30, 30, 30))
    small = font(SANS, 14)
    for i, line in enumerate(["Refrigerant R-454B  Charge 3.6 kg", "Rated voltage 400V 3~ 50Hz",
                              "Max operating pressure 4.2 MPa", "Year of manufacture 2025", "IP rating IPX4"]):
        d.text((180, 260 + i * 30), line, font=small, fill=(40, 40, 40))
    d.text((660, 560), "SN 7Q4-K2M-118", font=font(MONO, 13), fill=(30, 30, 30))
    return image


def bar_chart():
    image, d = canvas()
    f = font(SANS, 16)
    values = [("Jan", 412), ("Feb", 388), ("Mar", 455), ("Apr", 501), ("May", 563), ("Jun", 547), ("Jul", 498),
              ("Aug", 476)]
    d.text((120, 60), "Decode throughput by month (tok/min, thousands)", font=font(BOLD, 22), fill=(20, 20, 20))
    base, left, width = 660, 130, 80
    for i, (month, value) in enumerate(values):
        x = left + i * 100
        top = base - value
        d.rectangle([x, top, x + width, base], fill=(78, 121, 167))
        d.text((x + 22, top - 24), str(value), font=f, fill=(20, 20, 20))
        d.text((x + 24, base + 10), month, font=f, fill=(20, 20, 20))
    return image


def dialog():
    image, d = canvas((60, 64, 72))
    d.rectangle([212, 154, 812, 614], fill=(246, 246, 246), outline=(30, 30, 30))
    d.text((240, 176), "Export settings", font=font(BOLD, 24), fill=(20, 20, 20))
    f = font(SANS, 18)
    for i, (label, value) in enumerate([("Format", "GLB (binary)"), ("Scale", "1.0"), ("Apply modifiers", "Yes")]):
        d.text((250, 240 + i * 50), label, font=f, fill=(40, 40, 40))
        d.rectangle([470, 234 + i * 50, 770, 266 + i * 50], fill=(255, 255, 255), outline=(150, 150, 150))
        d.text((480, 240 + i * 50), value, font=f, fill=(40, 40, 40))
    buttons = {"Cancel": (444, 548, 554, 590), "Apply": (566, 548, 676, 590), "Save": (688, 548, 798, 590)}
    for name, box in buttons.items():
        d.rectangle(box, fill=(52, 120, 246) if name == "Save" else (225, 225, 225), outline=(120, 120, 120))
        d.text((box[0] + 30, box[1] + 10), name, font=f, fill=(255, 255, 255) if name == "Save" else (30, 30, 30))
    return image, buttons["Save"]


def toolbar():
    image, d = canvas()
    f = font(SANS, 15)
    names = ["New", "Open", "Save", "Undo", "Redo", "Import", "Export", "Render"]
    boxes = {}
    for i, name in enumerate(names):
        x = 60 + i * 115
        box = (x, 80, x + 96, 176)
        boxes[name] = box
        d.rounded_rectangle(box, radius=10, fill=(236, 239, 244), outline=(170, 176, 186))
        d.ellipse([x + 30, 92, x + 66, 128], fill=(90 + i * 18, 110, 160 - i * 10))
        d.text((x + 48 - len(name) * 4, 140), name, font=f, fill=(30, 30, 30))
    d.text((60, 260), "Viewport", font=font(BOLD, 20), fill=(120, 120, 120))
    d.rectangle([60, 290, 964, 720], outline=(200, 200, 200))
    return image, boxes["Export"]


def map_pins():
    image, d = canvas((226, 232, 214))
    for x in range(0, W, 64):
        d.line([x, 0, x, H], fill=(206, 214, 194))
    for y in range(0, H, 64):
        d.line([0, y, W, y], fill=(206, 214, 194))
    f = font(SANS, 16)
    pins = {"Trailhead": (120, 610), "Camp 1": (330, 470), "Camp 2": (520, 380), "Camp 3": (742, 214),
            "Summit": (880, 120), "Ranger hut": (210, 250)}
    target = None
    for name, (x, y) in pins.items():
        color = (210, 40, 40) if name == "Camp 3" else (50, 90, 170)
        d.ellipse([x - 10, y - 10, x + 10, y + 10], fill=color, outline=(20, 20, 20))
        d.text((x + 14, y - 10), name, font=f, fill=(20, 20, 20))
        if name == "Camp 3":
            target = (x - 12, y - 12, x + 14 + 60, y + 12)
    return image, target


def form():
    image, d = canvas()
    f = font(SANS, 18)
    d.text((300, 90), "Create account", font=font(BOLD, 28), fill=(20, 20, 20))
    fields = ["Full name", "Email", "Password", "Confirm password"]
    boxes = {}
    for i, name in enumerate(fields):
        y = 170 + i * 110
        d.text((300, y), name, font=f, fill=(40, 40, 40))
        box = (300, y + 30, 724, y + 74)
        boxes[name] = box
        d.rectangle(box, fill=(255, 255, 255), outline=(150, 150, 150))
    return image, boxes["Email"]


ASK = "Open it with your read tool (it is an image, {w} x {h} pixels)."
OCR = [
    ("vision-ocr-receipt-total", "receipts/grocery.png", receipt_grocery,
     "What is the receipt's TOTAL? End your answer with a line `TOTAL: <amount>`.", [r"(?m)^TOTAL:\s*\$?47\.83\s*$"],
     "TOTAL: 47.83"),
    ("vision-ocr-receipt-line", "receipts/hardware.png", receipt_hardware,
     "What does the line for the M8 hex bolts cost? End with a line `PRICE: <amount>`.", [r"(?m)^PRICE:\s*\$?6\.72\s*$"],
     "PRICE: 6.72"),
    ("vision-ocr-terminal-error", "screens/terminal.png", terminal,
     "The server failed to start. Quote the exact exception line. End with a line `ERROR: <the exception line>`.",
     [r"(?m)^ERROR:\s*`?ModuleNotFoundError: No module named 'fastapi_utils'`?\s*$"],
     "ERROR: ModuleNotFoundError: No module named 'fastapi_utils'"),
    ("vision-ocr-config-table", "screens/presets.png", config_table,
     "Which context and KV cache does `lumen-35b-a3b` use, and how many CPU MoE layers? End with a line "
     "`ROW: <context>, <kv>, <cpu moe>`.", [r"(?m)^ROW:\s*262,?144,\s*q8_0,\s*9\s*$"], "ROW: 262,144, q8_0, 9"),
    ("vision-ocr-serial", "photos/label.png", label_plate,
     "What serial number is printed on the unit's label? End with a line `SERIAL: <serial>`.",
     [r"(?m)^SERIAL:\s*(SN\s*)?7Q4-K2M-118\s*$"], "SERIAL: SN 7Q4-K2M-118"),
    ("vision-ocr-chart", "charts/throughput.png", bar_chart,
     "Which month has the highest bar, and what value is printed on it? End with a line `PEAK: <month> <value>`.",
     [r"(?m)^PEAK:\s*May\s+563\s*$"], "PEAK: May 563"),
]
GROUND = [
    ("vision-ground-dialog-save", "screens/dialog.png", dialog, "the Save button"),
    ("vision-ground-toolbar-export", "screens/toolbar.png", toolbar, "the Export tool (its whole tile)"),
    ("vision-ground-map-camp3", "maps/trail.png", map_pins, "the Camp 3 pin and its label together"),
    ("vision-ground-form-email", "screens/form.png", form, "the Email input field (the box you would type in)"),
]


def one_value(answer):
    """Every line that states the answer's field (``PRICE:``, also bulleted or bold), capturing its value: the
    grader requires one value across them (RT-005), so a list of candidates fails."""
    field = answer.split(":")[0]
    return rf"(?m)^[\s*_>#-]*{field}\s*:[*_\s]*(.*?)[*_\s]*$"


def write():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / ".git").exists():
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=OUT, check=True)
    (OUT / "README.md").write_text("# vision-set\n\nImages for the vision tasks; answers live in the tasks, not here.\n")
    boxes = {}
    for task_id, rel, draw, *_ in OCR:
        path = OUT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        draw().save(path)
    for task_id, rel, draw, _ in GROUND:
        path = OUT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        image, box = draw()
        image.save(path)
        boxes[task_id] = list(box)
    subprocess.run(["git", "add", "-A"], cwd=OUT, check=True)
    subprocess.run(["git", "-c", "user.name=llm-eval", "-c", "user.email=llm-eval@localhost", "commit", "-qm",
                    "vision-set: ten images", "--allow-empty"], cwd=OUT, check=True, capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=OUT, capture_output=True, text=True).stdout.strip()
    folder = TASK_DIR / "vision"
    folder.mkdir(parents=True, exist_ok=True)
    source = {"repo": str(OUT).replace(str(Path.home()), "~"), "commit": commit, "workdir": "."}
    base = {"kind": "workload", "category": "vision", "requires": ["image"], "source": source,
            "limits": {"turns": 1, "wall_seconds": 600}}
    for task_id, rel, _, question, patterns, answer in OCR:
        task = {"id": task_id, **base,
                "prompt": f"Task packet (vision-set, ~/Workspace/vision-set). Look at `{rel}`. {ASK.format(w=W, h=H)} "
                          f"{question}",
                "checks": [{"type": "answer", "all": patterns, "one_value": [one_value(answer)]}],
                "solution": {"answer": answer}}
        (folder / f"{task_id}.json").write_text(json.dumps(task, indent=2) + "\n")
    for task_id, rel, _, thing in GROUND:
        box = boxes[task_id]
        task = {"id": task_id, **base,
                "prompt": f"Task packet (vision-set, ~/Workspace/vision-set). Look at `{rel}`. {ASK.format(w=W, h=H)} "
                          f"Where is {thing}? Give its bounding box in the image's pixel coordinates (origin top "
                          "left), and end with a line `BOX: x1, y1, x2, y2`.",
                "checks": [{"type": "box", "target": box, "size": [W, H], "min_iou": 0.5}],
                "solution": {"answer": "BOX: " + ", ".join(str(v) for v in box)}}
        (folder / f"{task_id}.json").write_text(json.dumps(task, indent=2) + "\n")
    print(f"{len(OCR) + len(GROUND)} vision tasks at commit {commit[:12]}")


if __name__ == "__main__":
    write()
