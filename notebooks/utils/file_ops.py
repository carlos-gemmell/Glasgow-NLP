def corpus_to_array(src_fp, tgt_fp):
    lines = []
    with open(src_fp, "r", encoding="utf-8") as src_file, open(tgt_fp, "r", encoding="utf-8") as tgt_file:
        for src, tgt in zip(src_file, tgt_file):
            lines.append((src, tgt))
    return lines


def array_to_corpus(lines, src_out_file_name, tgt_out_file_name):
    with open(src_out_file_name, "w+", encoding="utf-8") as src_file, open(tgt_out_file_name, "w+", encoding="utf-8") as tgt_file:
        for src, tgt in lines:
            src_file.write(src)
            tgt_file.write(tgt)
    