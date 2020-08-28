import PyNomad
import threading
import queue
import sys
import time

class PyNomadAsync:

    def __init__(self, block_q, n_max_evals):
        self.n_max_evals = n_max_evals
        self.q = block_q
        #  self.x_return = None
        #  self.h_return = None
        self.t = None

    def bb(self, x, bb_out, id_t):
        try:
            while self.q.empty():
                continue
            self.q.get()
            dim = x.get_n()
            out = [ x.get_coord(4) ]
            g1 = sum([(x.get_coord(i)-1)**2 for i in range(dim)])-25
            out.append(g1)
            g2 = 25-sum([(x.get_coord(i)+1)**2 for i in range(dim)])
            out.append(g2)
            bb_out.put((id_t, out))
            self.q.task_done()
        except:
            print ("Unexpected error in bb()", sys.exc_info()[0])
            return -1
        return 1 # 1: success 0: failed evaluation

    def start(self):

        x0 = [0,0,0.71,0.51,0.51]
        lb = [-6,-6,-6,-6,-6]
        ub = [ 5,6,7,10,10]

        params = ['BB_OUTPUT_TYPE OBJ PB EB','MAX_BB_EVAL 100' , 'BB_MAX_BLOCK_SIZE 8','DISPLAY_STATS BBE BLK_EVA OBJ','DISABLE MODELS']
        #[ x_return , f_return , h_return, nb_evals , nb_iters ,  stopflag ] = 
        self.t = threading.Thread(target=PyNomad.optimize, args=(self.bb,x0,lb,ub,params,), daemon=True)
        self.t.start()
        #  print ('\n NOMAD outputs \n X_sol={} \n F_sol={} \n H_sol={} \n NB_evals={} \n NB_iters={} \n'.format(x_return,f_return,h_return,nb_evals,nb_iters))

queue = queue.Queue()
opt = PyNomadAsync(queue, 100)
opt.start()
for i in range(100):
    time.sleep(0.5)
    if opt.t.is_alive():
        queue.put("ppupu")
        queue.join()
