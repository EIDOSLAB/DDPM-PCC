from statistics import mean

def get_bpp(encodings_out_file):
    bpps = []
    file1 = open(encodings_out_file, 'r')
    for line in file1.readlines():
        l = line.strip()
        if 'bpp' in l:
            bpp = l.split('(')[-1].replace(' bpp)','')
            bpp = float(bpp)
            bpps.append(bpp)

    print(f'Founded {len(bpps)} values')
    return mean(bpps)

def get_bpp_with_memory(encodings_out_file, n_points = 2048):
    bpps = []
    file1 = open(encodings_out_file, 'r')
    for line in file1.readlines():
        l = line.strip()
        if 'bpp' in l:
            assert ' B' in l and 'size ' in l, f'Find line: {l} without memory size (B)'
            size = l.split(' B')[0].split('size ')[-1]
            size = float(size)*8
            bpp = size / n_points
            # bpp = float(bpp)
            bpps.append(bpp)
    
    print(f'Founded {len(bpps)} values')
    return mean(bpps)


def get_bpp_with_memory_draco(encodings_out_file, n_points = 2048):
    bpps = []
    file1 = open(encodings_out_file, 'r')
    for line in file1.readlines():
        l = line.strip()
        if 'Encoded size = ' in l:
            assert ' bytes' in l, f'Find line: {l} without memory size (B)'
            size = l.split(' bytes')[0].split('Encoded size = ')[-1]
            size = float(size)*8
            bpp = size / n_points
            # bpp = float(bpp)
            bpps.append(bpp)
    
    print(f'Founded {len(bpps)} values')
    return mean(bpps)


def get_bpp_with_entire_memory(encodings_out_file, n_points = 2048):
    bpps = []
    file1 = open(encodings_out_file, 'r')
    for line in file1.readlines():
        l = line.strip()
        if 'Total frame size ' in l:
            assert ' B' in l, f'Find line: {l} without memory size (B)'
            size = l.split('Total frame size ')[-1].replace(' B','')
            size = float(size)*8
            bpp = size / n_points
            # bpp = float(bpp)
            bpps.append(bpp)
    print(f'Founded {len(bpps)} values')
    return mean(bpps)

if __name__ == '__main__':

    for i in range(6):
        encodings_out_file = f'/home/ids/gspadaro/repos/PoitcloudsCompression/src/results/vq/mpeg/dummy_same_rate_merge/r0{i+1}/encodings.out'
        avg_bpp = get_bpp_with_memory(encodings_out_file)
        print(f'BPP (avg): {avg_bpp}')

    