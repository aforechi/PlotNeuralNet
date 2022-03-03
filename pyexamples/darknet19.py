
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

classes = 658
tamanho = 128

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    
    to_input( '../examples/fcn8s/cats.jpg', height=26, width=26 ),

    to_Conv("conv1", 448, 32, offset="(0,0,0)", to="(0,0,0)", height=tamanho, depth=tamanho, width=2 ),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", height=tamanho/2, depth=tamanho/2),

    to_Conv("conv2", 224, 64, offset="(1,0,0)", to="(pool1-east)", height=tamanho/2, depth=tamanho/2, width=4 ),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=tamanho/4, depth=tamanho/4),

    to_Conv("conv3a", 112, 128, offset="(1,0,0)", to="(pool2-east)", height=tamanho/4, depth=tamanho/4, width=6 ),
    to_Conv("conv3b", 112, 64, offset="(0.5,0,0)", to="(conv3a-east)", height=tamanho/4, depth=tamanho/4, width=4 ),
    to_Conv("conv3c", 112, 128, offset="(0.5,0,0)", to="(conv3b-east)", height=tamanho/4, depth=tamanho/4, width=6 ),
    to_Pool("pool3", offset="(0,0,0)", to="(conv3c-east)", height=tamanho/8, depth=tamanho/8),

    to_Conv("conv4a", 56, 256, offset="(1,0,0)", to="(pool3-east)", height=tamanho/8, depth=tamanho/8, width=10 ),
    to_Conv("conv4b", 56, 128, offset="(0.5,0,0)", to="(conv4a-east)", height=tamanho/8, depth=tamanho/8, width=6 ),
    to_Conv("conv4c", 56, 256, offset="(0.5,0,0)", to="(conv4b-east)", height=tamanho/8, depth=tamanho/8, width=10 ),
    to_Pool("pool4", offset="(0,0,0)", to="(conv4c-east)", height=tamanho/16, depth=tamanho/16),

    to_Conv("conv5a", 28, 512, offset="(1,0,0)", to="(pool4-east)", height=tamanho/16, depth=tamanho/16, width=15 ),
    to_Conv("conv5b", 28, 256, offset="(0.5,0,0)", to="(conv5a-east)", height=tamanho/16, depth=tamanho/16, width=10 ),
    to_Conv("conv5c", 28, 512, offset="(0.5,0,0)", to="(conv5b-east)", height=tamanho/16, depth=tamanho/16, width=15 ),
    to_Conv("conv5d", 28, 256, offset="(0.5,0,0)", to="(conv5c-east)", height=tamanho/16, depth=tamanho/16, width=10 ),
    to_Conv("conv5e", 28, 512, offset="(0.5,0,0)", to="(conv5d-east)", height=tamanho/16, depth=tamanho/16, width=15 ),
    to_Pool("pool5", offset="(0,0,0)", to="(conv5e-east)", height=tamanho/32, depth=tamanho/32),

    to_Conv("conv6a", 14, 1024, offset="(1,0,0)", to="(pool5-east)", height=tamanho/32, depth=tamanho/32, width=30 ),
    to_Conv("conv6b", 14, 512, offset="(0.5,0,0)", to="(conv6a-east)", height=tamanho/32, depth=tamanho/32, width=15 ),
    to_Conv("conv6c", 14, 1024, offset="(0.5,0,0)", to="(conv6b-east)", height=tamanho/32, depth=tamanho/32, width=30 ),
    to_Conv("conv6d", 14, 512, offset="(0.5,0,0)", to="(conv6c-east)", height=tamanho/32, depth=tamanho/32, width=15 ),
    to_Conv("conv6e", 14, 1024, offset="(0.5,0,0)", to="(conv6d-east)", height=tamanho/32, depth=tamanho/32, width=30 ),
    to_Pool("pool6", offset="(0,0,0)", to="(conv6e-east)", height=tamanho/64, depth=tamanho/64),

    # todas as outras convoluções utilizam batchnorm e função de ativação leaky
    to_Conv("conv7", 14, classes, offset="(1,0,0)", to="(pool6-east)", height=tamanho/64, depth=tamanho/64, width=20, caption="linear" ),
    to_AvgPool("pool7", offset="(0,0,0)", to="(conv7-east)", height=tamanho/64, depth=tamanho/64, caption="AVG"),

    to_SoftMax("soft1", classes ,"(2,0,0)", "(pool7-east)", caption="SOFT"  ),
 
    to_end()
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
