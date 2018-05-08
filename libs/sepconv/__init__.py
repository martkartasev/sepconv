
from os import system
from os.path import exists, join

sepconv_dir = __path__[0]
if not exists(join(sepconv_dir, '_ext')):
    print('sepconv._ext not found, running installer...')
    system(join(sepconv_dir, 'install.bash'))
