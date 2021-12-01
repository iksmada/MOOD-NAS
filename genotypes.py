from collections import namedtuple


class Genotype(namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')):
    def __eq__(self, other):
        return self.normal == other.normal and list(self.normal_concat) == list(other.normal_concat) and \
               self.reduce == other.reduce and list(self.reduce_concat) == list(other.reduce_concat)


PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

PC_DARTS_cifar = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(
    normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

# empirical weight genotypes
l1_6e5_l2_0 = Genotype(
    normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0),
            ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3),
            ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
l1_1e4_l2_0 = Genotype(
    normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0),
            ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
l1_1e4_l2_3e4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2),
            ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
l1_6e5_l2_3e4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1),
            ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
l1_0_l2_1e4 = Genotype(
    normal=[('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1),
            ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
l1_0_l2_1e3 = Genotype(
    normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('max_pool_3x3', 1), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0),
            ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))
half_l1_1e4_l2_3e4 = Genotype(
    normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3),
            ('max_pool_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

# weight estimator genotypes
l2_loss_2e01 = Genotype(
    normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 2),
            ('max_pool_3x3', 3), ('max_pool_3x3', 3), ('max_pool_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l2_loss_9e04 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

l2_loss_7e03 = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 3),
            ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 2),
            ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l2_loss_23e04 = Genotype(
    normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1),
            ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

l2_loss_3e02 = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l2_loss_1e01 = Genotype(
    normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l2_loss_429e05 = Genotype(
    normal=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 3),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

l2_loss_13e04 = Genotype(
    normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 3), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 1)], reduce_concat=range(2, 6))

l2_loss_196e05 = Genotype(
    normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2),
            ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2),
            ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

l2_loss_8e02 = Genotype(
    normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l2_loss_1e02 = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l2_loss_425e05 = Genotype(
    normal=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3),
            ('sep_conv_3x3', 2), ('sep_conv_5x5', 4), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

l2_loss_14e04 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

l2_loss_191e05 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 2),
            ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1),
            ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))

l2_loss_3e03 = Genotype(
    normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3),
            ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l2_loss_6e02 = l2_loss_1e01

l2_loss_7e04 = Genotype(
    normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 2), ('skip_connect', 1),
            ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

l1_loss_2e01_l2_3e04 = Genotype(
    normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 2),
            ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

l1_loss_1e04_l2_3e04 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 2),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3),
            ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))

l1_loss_3e06_l2_3e04 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_3x3', 2),
            ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1),
            ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

l1_loss_8e04_l2_3e04 = Genotype(
    normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1),
            ('dil_conv_3x3', 2), ('dil_conv_5x5', 2), ('sep_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('avg_pool_3x3', 4), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_7e06_l2_3e04 = Genotype(
    normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1),
            ('skip_connect', 3), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0)], reduce_concat=range(2, 6))

l1_loss_4e06_l2_3e04 = Genotype(
    normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0),
            ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_5x5', 3), ('dil_conv_3x3', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))

l1_loss_4e03_l2_3e04 = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 2),
            ('dil_conv_3x3', 3), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_1e02_l2_3e04 = Genotype(
    normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 3),
            ('avg_pool_3x3', 2), ('avg_pool_3x3', 4), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_2e03_l2_3e04 = Genotype(
    normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_2e04_l2_3e04 = Genotype(
    normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 3),
            ('avg_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_8e03_l2_3e04 = Genotype(
    normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 0), ('avg_pool_3x3', 3),
            ('avg_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_2e02_l2_3e04 = Genotype(
    normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 2), ('dil_conv_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_2e01 = Genotype(
    normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1),
            ('skip_connect', 0), ('avg_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2),
            ('max_pool_3x3', 3), ('avg_pool_3x3', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))

l1_loss_6e05 = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_5x5', 2),
            ('skip_connect', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

l1_loss_2e04 = Genotype(
    normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3),
            ('dil_conv_3x3', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2),
            ('avg_pool_3x3', 3), ('skip_connect', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

l1_loss_1e03 = Genotype(
    normal=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('max_pool_3x3', 3),
            ('dil_conv_5x5', 1), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 3),
            ('sep_conv_3x3', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_5e03 = Genotype(
    normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3),
            ('dil_conv_5x5', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_1e02 = Genotype(
    normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_3e03 = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 2),
            ('dil_conv_3x3', 3), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_4e04 = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 3),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

l1_loss_9e03 = Genotype(
    normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
