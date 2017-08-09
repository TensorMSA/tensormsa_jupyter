class Config():

    edim = 150
    lindim = 75
    nhop = 6
    mem_size = 100
    batch_size = 128
    nepoch = 1
    init_lr = 0.01
    init_hid = 0.1
    init_std = 0.05
    max_grad_norm = 50
    data_dir = 'data'
    checkpoint_dir = 'checkpoints'
    vector_dir = 'vectorpoints'
    data_name = 'ptb'
    is_test = False
    show = False