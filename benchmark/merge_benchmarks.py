# A list of filenames and the column to take
files = [
    ["search_GPU_128.csv", 1],
    ["search_GPU_256.csv", 1],
    ["search_GPU_512.csv", 1],
    ["search_GPU_1024.csv", 1],
]
nb_rows = 1001
out_file = "GPU_search_block_size_comparison.csv"
first_line = "128, 256, 512, 1024"

##############################################################################

values = [[] for _ in range(1001)]
for filename, column in files:
    with open(filename, 'r') as fread:
        i = 0
        for line in fread:
            val = line.strip().split(",")[column]
            values[i].append(val)
            i += 1

with open(out_file, "w") as fwrite:
    fwrite.write(first_line + "\n")
    fwrite.write("\n".join([",".join(val for val in vl) for vl in values]))
