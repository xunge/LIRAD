from tqdm import tqdm
import json


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    data_path = '/home/jiang/dataset/data_merge/data_gcc_clang_llvmir_merge-all.jsonl'
    output_path = '/home/jiang/dataset/LIRAD_input/decompile_asm_gcc_clang_ll_clang_src.jsonl'
    data_list = read_jsonl(data_path)
    opt_list = ['O0', 'O1', 'O2', 'O3']
    before_decompile_src = f"# This is the assembly code:\n"
    after_decompile_src = "\n# What is the source code?\n"
    before_align_src = f'# Convert the assembly code to LLVM-IR, then write the equivalent C function.\n'
    before_align_target = f'# Here is the LLVM-IR code:\n'
    mid_align_target = f'\n# And here is the equivalent C function:\n'
    output_list = []
    for item_dict in tqdm(data_list):
        src = item_dict['function_def']
        for opt in opt_list:
            if f'-{opt}_gcc' not in item_dict or f'-{opt}_clang' not in item_dict or f'-{opt}_ll' not in item_dict:
                continue
            opt_asm_gcc = item_dict[f'-{opt}_gcc']
            opt_asm_clang = item_dict[f'-{opt}_clang']
            opt_ll = item_dict[f'-{opt}_ll']
            # output_list.append({'source': before_decompile_src + opt_asm_gcc + after_decompile_src, 'target': src, 'category': 'decompile'})
            output_list.append({'source': before_decompile_src + opt_asm_clang + after_decompile_src, 'target': src, 'category': 'decompile'})
            output_list.append({'source': before_align_src + opt_asm_clang, 'target': before_align_target + opt_ll + mid_align_target + src, 'category': 'align'})
    write_jsonl(output_list, output_path)

