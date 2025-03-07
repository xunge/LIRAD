import os


def concatenate_text_files(input_folder, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for root, _, files in os.walk(input_folder):
                files.sort()
                for file in files:
                    if file.endswith('.jsonl'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
    except Exception as e:
        print(e)


if __name__ == '__main__':
    input_folder = '/home/jiang/dataset/data_gcc_llvmir_clang_multi/'
    output_file = '/home/jiang/dataset/data_merge/data_gcc_clang_llvmir_merge-all.jsonl'
    concatenate_text_files(input_folder, output_file)
