import sys
import re

# ============================================================
# 全局配置
# ============================================================

# GBuffer layout
GBUFFER_OUTPUTS = {
    "BaseColor": ["o3.x", "o3.y", "o3.z"],
    "Normal": ["o1.x", "o1.y"],
    "Mask": ["o2.x", "o2.y", "o2.z"],
}


# 传播依赖指令
PROPAGATE_OPS = {
    "mov",
    "mov_sat",
    "movc",
    "movc_sat",
    "mul",
    "mul_sat",
    "add",
    "mad",
    "mad_sat",
    "div",
    "max",
    "min",
    "dp2",
    "dp3",
    "sqrt",
    "rsq",
}


UV_UNKNOWN = "unknown"
UV_MESH = "mesh_uv"
UV_SCREEN = "screen_uv"


# ============================================================
# 文件读取 + 指令提取
# ============================================================

def read_shader_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def extract_instruction_lines(lines):
    """提取指令行(idx, code)"""
    instructions = []

    for line in lines:
        match = re.match(r"\s*(\d+):\s*(.*)$", line)
        if not match:
            continue
            
        idx = int(match.group(1))
        code = match.group(2).strip()

        instructions.append((idx, code))
    
    return instructions


# ============================================================
# 指令解析
# ============================================================

def split_args(arg_text):
    """按逗号分割指令参数"""
    return [part.strip() for part in arg_text.split(",")]


def parse_instruction(code):
    """拆分 op 和 args"""
    if not code:
        return None
    
    # 解析 op
    parts = code.split(None, 1)
    op_full = parts[0]

    # sample_b(texture2d)(...)只保留sample_b
    op = op_full.split("(")[0]

    # 解析 args
    if len(parts) == 1:
        args = []
    else:
        args = split_args(parts[1])
    
    return {
        "op": op,
        "args": args,
        "code": code
    }


# ============================================================
# 寄存器工具
# ============================================================

def clean_token(token):
    """
    清理不影响依赖关系的符号
    如 abs(r0.x) -> r0.x
       -r5.xyz -> r5.xyz
    """
    token = token.strip()

    while token.startswith("-"):
        token = token[1:].strip()
    
    if token.startswith("abs(") and token.endswith(")"):
        token = token[4:-1].strip()
    
    return token


def expand_dst_components(token):
    """展开 dst 分量"""
    token = token.strip()

    if "." not in token:
        return [token + "." + c for c in "xyzw"]
    
    reg, comps = token.split(".", 1)
    return [reg + "." + c for c in comps]


def expand_src_components(token, count):
    """"按dst分量数量展开src分量"""
    token = clean_token(token)

    match = re.match(r"(r\d+|o\d+|v\d+)(?:\.([xyzw]+))?$", token)
    if not match:
        return []
    
    reg = match.group(1)
    comps = match.group(2)

    if comps is None:
        comps = "xyzw"
    
    result = []

    for i in range(count):
        if i < len(comps):
            c = comps[i]
        else:
            c = comps[-1]

        result.append(reg + "." + c)

    return result


def get_register_name(token):
    """从 r3.xy / -r3.xyz / abs(r3.x) 中取出 r3"""
    token = clean_token(token)

    match = re.match(r"(r\d+|o\d+|v\d+)(?:\.|$)", token)
    if not match:
        return None
    
    return match.group(1)


def instruction_uses_register(inst, reg):
    """判断一条指令的源参数里是否使用了某个寄存器"""
    if reg is None:
        return False
    
    if len(inst["args"]) < 2:
        return False
    
    for arg in inst["args"][1:]:
        used_reg = get_register_name(arg)

        if used_reg == reg:
            return True
    
    return False


# ============================================================
# UV 类型传播
# ============================================================

def init_coord_type():
    """初始化 UV 类型"""
    coord_type = {}

    coord_type["v3.x"] = UV_MESH
    coord_type["v3.y"] = UV_MESH

    for c in "xyzw":
        coord_type["v6." + c] = UV_SCREEN
    
    return coord_type


def merge_coord_types(types):
    """合并 UV 类型"""
    useful = set(t for t in types if t != UV_UNKNOWN)

    if len(useful) == 1:
        return next(iter(useful))
    
    return UV_UNKNOWN


def collect_source_coord_types(src_args, dst_count, coord_type):
    """收集源参数中的 UV 类型"""
    types = []

    for arg in src_args:
        src_comps = expand_src_components(arg, dst_count)

        for comp in src_comps:
            types.append(coord_type.get(comp, UV_UNKNOWN))
    
    return types


def process_coord_instruction(inst, coord_type):
    """传播 UV 类型"""
    if len(inst["args"]) < 2:
        return
    
    if inst["op"] not in PROPAGATE_OPS:
        return
    
    dst = inst["args"][0]
    dst_comps = expand_dst_components(dst)

    if inst["op"] in ("movc", "movc_sat"):
        src_args = inst["args"][2:]
    else:
        src_args = inst["args"][1:]
    
    src_types = collect_source_coord_types(src_args, len(dst_comps), coord_type)
    result_type = merge_coord_types(src_types)

    for comp in dst_comps:
        coord_type[comp] = result_type


def get_sample_uv_type(inst, coord_type):
    """获取 sample 指令的 UV 类型"""
    if len(inst["args"]) < 2:
        return UV_UNKNOWN
    
    uv_arg = inst["args"][1]
    
    # 仅处理 texture2d
    uv_comps = expand_src_components(uv_arg, 2)

    types = []
    for comp in uv_comps:
        types.append(coord_type.get(comp, UV_UNKNOWN))
    
    return merge_coord_types(types)


def record_texture_uv_type(slot, uv_type, tex_uv_type):
    """记录 texture 的 UV 类型"""
    old_type = tex_uv_type.get(slot, UV_UNKNOWN)

    if old_type == UV_UNKNOWN:
        tex_uv_type[slot] = uv_type
    elif uv_type == UV_UNKNOWN:
        return
    elif old_type == uv_type:
        return
    else:
        tex_uv_type[slot] = UV_UNKNOWN


# ============================================================
# texture 依赖传播
# ============================================================

def find_texture_slot(args):
    """从指令参数中找到 texture slot"""
    for arg in args:
        match = re.match(r"(t\d+)(?:\.|$)", arg)
        if match:
            return match.group(1)
    
    return None


def collect_source_deps(src_args, dst_count, reg_deps):
    """收集源参数中的 texture 依赖"""
    deps = set()

    for arg in src_args:
        src_comps = expand_src_components(arg, dst_count)

        for comp in src_comps:
            deps.update(reg_deps.get(comp, set()))
    
    return deps


def process_sample_instruction(idx, inst, reg_deps, coord_type, tex_uv_type, sample_records):
    """
    处理 sample 指令
    1. 记录 dst 的 texture 依赖
    2. 记录 texture 的 UV 类型
    3. 记录 sample 事件，后续判断 texture role
    """
    if len(inst["args"]) < 3:
        return
    
    dst = inst["args"][0]
    uv_arg = inst["args"][1]
    slot = find_texture_slot(inst["args"])

    if slot is None:
        return
    
    # 记录 texture 依赖
    for comp in expand_dst_components(dst):
        reg_deps[comp] = {slot}
    
    # 记录该 texture 的 UV 类型
    uv_type = get_sample_uv_type(inst, coord_type)
    record_texture_uv_type(slot, uv_type, tex_uv_type)

    # 记录 sample 事件
    sample_records.append({
        "line": idx,
        "op": inst["op"],
        "slot": slot,
        "dst": dst,
        "uv_arg": uv_arg,
        "uv_type": uv_type,
    })


def process_propagate_instruction(inst, reg_deps):
    """处理依赖传播指令"""
    if len(inst["args"]) < 2:
        return
    
    dst = inst["args"][0]
    dst_comps = expand_dst_components(dst)

    if inst["op"] in ("movc", "movc_sat"):
        src_args = inst["args"][2:]
    else:
        src_args = inst["args"][1:]
    
    deps = collect_source_deps(src_args, len(dst_comps), reg_deps)

    for comp in dst_comps:
        reg_deps[comp] = set(deps)


# ============================================================
# 分析流程
# ============================================================

def analyze_instructions(instructions):
    """顺序扫描指令"""
    reg_deps = {}
    coord_type = init_coord_type()
    tex_uv_type = {}
    sample_records = []

    for idx, code in instructions:
        inst = parse_instruction(code)
        if inst is None:
            continue

        process_coord_instruction(inst, coord_type)

        if inst["op"].startswith("sample"):
            process_sample_instruction(
                idx,
                inst,
                reg_deps,
                coord_type,
                tex_uv_type,
                sample_records
            )
        elif inst["op"] in PROPAGATE_OPS:
            process_propagate_instruction(inst, reg_deps)

    return reg_deps, coord_type, tex_uv_type, sample_records


def collect_output_deps(reg_deps):
    """收集最终 GBuffer 输出的 texture 依赖"""
    output_deps = {}

    for semantic, outputs in GBUFFER_OUTPUTS.items():
        deps = set()

        for comp in outputs:
            deps.update(reg_deps.get(comp, set()))

        output_deps[semantic] = deps

    return output_deps


def get_or_create_slot_info(slot_infos, slot):
    """获取一个 texture slot 的汇总信息"""
    if slot not in slot_infos:
        slot_infos[slot] = {
            "sample_count": 0,
            "uv_types": set(),
            "sample_dsts": [],
            "outputs": set(),

            "is_normal_like": False,
            "normal_evidence": [],
        }
    
    return slot_infos[slot]


def build_slot_infos(sample_records, output_deps):
    """构建 texture slot 的汇总信息"""
    slot_infos = {}

    # 从 sample_records 汇总采样信息
    for record in sample_records:
        slot = record["slot"]
        info = get_or_create_slot_info(slot_infos, slot)

        info["sample_count"] += 1
        info["uv_types"].add(record["uv_type"])

        if record["dst"] not in info["sample_dsts"]:
            info["sample_dsts"].append(record["dst"])
    
    # 从 output_deps 反查每个 slot 影响的输出
    for semantic, deps in output_deps.items():
        for slot in deps:
            info = get_or_create_slot_info(slot_infos, slot)
            info["outputs"].add(semantic)

    # set 转为 list
    for info in slot_infos.values():
        info["uv_types"] = sorted(info["uv_types"])
        info["outputs"] = sorted(info["outputs"])
    
    return slot_infos


def detect_normal_like_for_sample(record, instructions, window_size=7):
    """
    检测一次 sample 是否像 normal
        第一次 remap 必须用 sample dst register
        后续在小窗口内按顺序找 dp2/dp3 和 sqrt/rsq
    """
    sample_line = record["line"]
    sample_reg = get_register_name(record["dst"])

    stage = 0
    evidence = []

    for idx, code in instructions:
        if idx <= sample_line:
            continue
            
        if idx > sample_line + window_size:
            break

        inst = parse_instruction(code)
        if inst is None:
            continue

        op = inst["op"]

        # stage 0: 等待 normal remap
        # 必须用 sample dst register
        if stage == 0:
            if op in ("mad", "mad_sat", "mul", "mul_sat", "add"):
                if not instruction_uses_register(inst, sample_reg):
                    continue

                stage = 1
                evidence.append("line {}: remap op {}".format(idx, op))
            continue

        # stage 1: 等待长度计算
        if stage == 1:
            if op in ("dp2", "dp3"):
                stage = 2
                evidence.append("line {}: vector length op {}".format(idx, op))
            continue
        
        # stage 2: 等待 z 重建 / normalize
        if stage == 2:
            if op in ("sqrt", "rsq"):
                stage = 3
                evidence.append("line {}: reconstruct op {}".format(idx, op))
                break
    
    is_normal_like = stage == 3

    return is_normal_like, evidence


def detect_normal_like_slots(slot_infos, sample_records, instructions):
    """检测每个 texture slot 是否像 normal"""
    for record in sample_records:
        slot = record["slot"]

        is_normal_like, evidence = detect_normal_like_for_sample(record, instructions)

        if not is_normal_like:
            continue
        
        info = get_or_create_slot_info(slot_infos, slot)
        info["is_normal_like"] = True
        info["normal_evidence"].extend(evidence)


def choose_mesh_uv_slot(deps, tex_uv_type):
    """从一组 texture 依赖中选择 mesh_uv 的 slot"""
    mesh_slots = []

    for slot in deps:
        if tex_uv_type.get(slot) == UV_MESH:
            mesh_slots.append(slot)
    
    if len(mesh_slots) == 1:
        return mesh_slots[0]
    
    return None


def classify_material_slots(output_deps, tex_uv_type):
    """根据 GBuffer 输出依赖，得到 BaseColor/Normal/Mask 对应的 slot"""
    slots = {}

    for semantic, deps in output_deps.items():
        slots[semantic] = choose_mesh_uv_slot(deps, tex_uv_type)
    
    return slots


# ============================================================
# Debug 打印
# ============================================================

def texture_slot_sort_key(slot):
    match = re.match(r"t(\d+)$", slot)

    if match:
        return int(match.group(1))
    
    return 9999


def print_reg_deps(reg_deps):
    """打印当前所有寄存器分量的 texture 依赖"""
    print("Register dependencies:")

    for key in sorted(reg_deps.keys()):
        print("  {} <- {}".format(key, sorted(reg_deps[key])))


def print_output_deps(output_deps):
    """打印最终 GBuffer 输出的 texture 依赖"""
    print("GBuffer output dependencies:")

    for semantic, deps in output_deps.items():
        print("  {}: {}".format(semantic, sorted(deps)))


def print_coord_type(coord_type):
    """打印非 unknown 的 UV 类型"""
    print("Coord types:")

    for key in sorted(coord_type.keys()):
        if coord_type[key] != UV_UNKNOWN:
            print("  {} <- {}".format(key, coord_type[key]))


def print_tex_uv_type(tex_uv_type):
    """打印每个 texture slot 的采样 UV 类型"""
    print("Texture UV types:")

    for slot in sorted(tex_uv_type.keys()):
        print("  {} <- {}".format(slot, tex_uv_type[slot]))


def print_material_slots(slots):
    """打印最终识别出的材质 slot。"""
    print("Material slots:")

    for semantic in ["BaseColor", "Normal", "Mask"]:
        slot = slots.get(semantic)

        if slot is None:
            print("  {}: <not found or ambiguous>".format(semantic))
        else:
            print("  {}: {}".format(semantic, slot))


def print_sample_records(sample_records):
    """打印所有 sample 事件"""
    print("Sample records:")

    for record in sample_records:
        print(
            "  line {}: {} <- {}, uv={}, uv_type={}".format(
                record["line"],
                record["dst"],
                record["slot"],
                record["uv_arg"],
                record["uv_type"]
            )
        )


def print_slot_infos(slot_infos):
    """打印每个 texture slot 的汇总信息"""
    print("Slot infos:")

    for slot in sorted(slot_infos.keys(), key=texture_slot_sort_key):
        info = slot_infos[slot]

        print("  {}:".format(slot))
        print("    sample_count: {}".format(info["sample_count"]))
        print("    uv_types: {}".format(info["uv_types"]))
        print("    sample_dsts: {}".format(info["sample_dsts"]))
        print("    outputs: {}".format(info["outputs"]))
        print("    is_normal_like: {}".format(info["is_normal_like"]))
        print("    normal_evidence: {}".format(info["normal_evidence"]))


# ============================================================
# 命令行入口
# ============================================================

def main():
    if len(sys.argv) != 2:
        print("Usage: python shader_analyzer.py <ps_asm.txt>")
        return
    
    path = sys.argv[1]

    lines = read_shader_lines(path)
    instructions = extract_instruction_lines(lines)

    reg_deps, coord_type, tex_uv_type, sample_records = analyze_instructions(instructions)
    output_deps = collect_output_deps(reg_deps)
    slot_infos = build_slot_infos(sample_records, output_deps)
    detect_normal_like_slots(slot_infos, sample_records, instructions)
    slots = classify_material_slots(output_deps, tex_uv_type)

    #print()
    #print_reg_deps(reg_deps)
    
    #print()
    #print_coord_type(coord_type)
    
    #print()
    #print_tex_uv_type(tex_uv_type)

    #print()
    #print_sample_records(sample_records)

    print()
    print_slot_infos(slot_infos)
    
    print()
    print_output_deps(output_deps)

    print()
    print_material_slots(slots)


if __name__ == "__main__":
    main()
