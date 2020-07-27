import PyNomad

# This example of blackbox function is for a single process
# The blackbox output must be put in the Eval_Point passed as argument
def bb(x):
    dim = x.get_n()
    vmin = 10000
    v = [ 0 , 0 , 0 ]
    
    t0 = int(x.get_coord(0))
    v0 = x.get_coord(1)
    t1 = int(x.get_coord(2))
    v1 = x.get_coord(3)
    
    v[t0] = v0
    v[t1] = v1
    
    if v0 < vmin:
        vmin = v0
    
    if v1 < vmin:
        vmin = v1

    vt = v[0] + v[1] + v[2]
    h = vt - 10000

    if h <= 0 and vmin >=1:
        vt2 = vt*vt
        rev = v[0]* 0.0891 + v[1] * 0.2137 + v[2] * 0.2346
        risk = 0.01 * (v[0]/vt)*(v[0]/vt) + \
               0.05 * (v[1]/vt)*(v[1]/vt) + \
               0.09 * (v[2]/vt)*(v[2]/vt) +  \
               0.02 * (v[0]*v[1]/vt2) + \
               0.02 * (v[0]*v[2]/vt2) + \
               0.10 * (v[1]*v[2]/vt2)

        a = (risk-0.01)*100/0.08
        b = (rev-891)*100/1455

        f = pow( a*a + (100-b)*(100-b), 0.5 )
    else:
        f = 145

    x.set_bb_output(0, h )
    x.set_bb_output(1,1-vmin)
    x.set_bb_output(2,f)
    return 1 # 1: success 0: failed evaluation

x0 = [0,100,1,100]
lb = []
ub=[]

params = ['BB_INPUT_TYPE (C R C R)','NEIGHBORS_EXE "$python ./neighbors.py"','BB_OUTPUT_TYPE EB EB OBJ','MAX_BB_EVAL 500','F_TARGET 0.0','LOWER_BOUND ( - 0.0 - 0.0 )','UPPER_BOUND ( 2 10000 - 10000 )']

[ x_return , f_return , h_return, nb_evals , nb_iters ,  stopflag ] = PyNomad.optimize(bb,x0,lb,ub,params)
print ('\n NOMAD outputs \n X_sol={} \n F_sol={} \n H_sol={} \n NB_evals={} \n NB_iters={} \n'.format(x_return,f_return,h_return,nb_evals,nb_iters))
