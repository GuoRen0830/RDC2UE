import sys
import re

# GBuffer layout
GBUFFER_OUTPUTS = {
    "BaseColor": ["o3.x", "o3.y", "o3.z"],
    "Normal": ["o1.x", "o1.y"],
    "Mask": ["o2.x", "o2.y", "o2.z"],
}


def read_shader_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def extract_instruction_lines(lines):
    instructions = []

    for line in lines:
        match = re.match(r"\s*(\d+):\s*(.*)$", line)
        if not match:
            continue
            
        idx = int(match.group(1))
        code = match.group(2).strip()

        instructions.append((idx, code))
    
    return instructions


def split_args(arg_text):
    return [part.strip() for part in arg_text.split(",")]


def parse_instruction(code):
    if not code:
        return None
    
    # 解析 op
    parts = code.split(None, 1)
    op_full = parts[0]

    op = op_full.split("(")[0]

    # 解析 args
    if len(parts) == 1:
        args = []
    else:
        args = split_args(parts[1])
    
    return {
        "op": op,
        "args": args,
        "Code": code
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python shader_analyzer.py <ps_asm.txt>")
        return
    
    path = sys.argv[1]
    lines = read_shader_lines(path)
    instructions = extract_instruction_lines(lines)

    print("Parsed samples / outputs:")

    for idx, code in instructions:
        inst = parse_instruction(code)
        if inst is None:
            continue

        if inst["op"].startswith("sample") or any(arg.startswith("o") for arg in inst["args"]):
            print("  {}: op={}, args={}".format(idx, inst["op"], inst["args"]))


if __name__ == "__main__":
    main()
