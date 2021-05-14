import sys
import genotypes
from graphviz import Digraph


def plot(gen, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(gen) % 2 == 0
    steps = len(gen) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = gen[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename=filename, directory="image", cleanup=True, view=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)
    genotype_name = sys.argv[1]
    if genotype_name != "all":
        try:
            genotype = genotypes.__dict__[genotype_name]
        except KeyError:
            print("{} is not specified in genotypes.py".format(genotype_name))
            sys.exit(1)

        plot(genotype.normal, "normal_cell_%s" % genotype_name)
        plot(genotype.reduce, "reduction_cell_%s" % genotype_name)
    else:
        # skip declarations on genotype.py before DARTS_V1
        skip = True
        for key, genotype in genotypes.__dict__.items():
            if key == "DARTS_V1":
                skip = False

            if not skip and isinstance(genotype, genotypes.Genotype):
                plot(genotype.normal, "normal_cell_%s" % key)
                plot(genotype.reduce, "reduction_cell_%s" % key)
