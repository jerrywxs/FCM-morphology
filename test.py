file_path = 'record.txt'

with open(file_path, 'r') as f:
    pre_lis = []
    rec_lis = []
    f_lis = []
    for line in f.readlines():
        if 'Image' in line:
            print(line)
        elif 'precision' in line:
            pre_lis.append(float(line.split()[-1]))
        elif 'recall' in line:
            rec_lis.append(float(line.split()[-1]))
        else:
            if line.split()[-1] == 'nan':
                f_lis.append(0.0)
            else:
                f_lis.append(float(line.split()[-1]))
            # print(float(line.split()[-1]))

    f_lis = sorted(f_lis)[2:]
    print(f_lis)
    print(len(f_lis), sum(f_lis) / len(f_lis))
    index_lis = [i for i in range(1, len(f_lis)+1)]
    print(pre_lis[9], rec_lis[24])
    print((sum(pre_lis)) / 52)
    print((sum(rec_lis)) / 52)
    print([i for i, _ in sorted(zip(index_lis, f_lis), key=lambda x:x[1])])
    print([i for _, i in sorted(zip(index_lis, f_lis), key=lambda x: x[1])])
