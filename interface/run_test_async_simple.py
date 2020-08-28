import PyNomad
import threading
import queue
import time

class PyNomadAsync:

    def __init__(self, block_q, n_max_evals):
        self.n_max_evals = n_max_evals
        self.q = block_q
        #  self.x_return = None
        #  self.h_return = None
        self.t = None

    def bb(self, x):
        while self.q.empty():
            continue
        self.q.get()
        dim = x.get_n()
        f = sum([x.get_coord(i)**2 for i in range(dim)])
        x.set_bb_output(0, f )
        self.q.task_done()
        return 1 # 1: success 0: failed evaluation

    def start(self):
        x0 = [0.71,0.51,0.51]
        lb = [-1,-1,-1]
        ub=[]

        params = ['BB_OUTPUT_TYPE OBJ','MAX_BB_EVAL 100','UPPER_BOUND * 1']

        #[ x_return , f_return , h_return, nb_evals , nb_iters ,  stopflag ] = 
        self.t = threading.Thread(target=PyNomad.optimize, args=(self.bb,x0,lb,ub,params,), daemon=True)
        self.t.start()
        #  print ('\n NOMAD outputs \n X_sol={} \n F_sol={} \n H_sol={} \n NB_evals={} \n NB_iters={} \n'.format(x_return,f_return,h_return,nb_evals,nb_iters))

queue = queue.Queue()
opt = PyNomadAsync(queue, 100)
opt.start()
for i in range(100):
    #  time.sleep(1)
    if opt.t.is_alive():
        queue.put("ppupu")
        queue.join()
