import numpy as np

def read_ela_sb(file):
    """ Used to read ELA data (for Spitsbergen) """
    with open(file, "r") as data:
        lines = data.read().split("\n")
        n = len(lines)
        x = np.zeros([n, 2], dtype=np.float)
        for i in range(n):
            line = lines[i].split("\t")
            x[i,0] = line[0]
            x[i,1] = line[1]
    return x


if __name__ == "__main__":
    data = read_ela_sb("ELA_Spitsbergen.txt")
    print(np.shape(data))
    t = data[:,0]
    E = data[:,1]
    print(np.min(t), np.max(t))
    print(np.min(E), np.max(E), np.mean(E), np.std(E))
