import PyNomad
import sys

#---------------------------------------------------------------------
#THIS EXAMPLES DOES NOT WORK ON DEFAULT WINDOWS ANACONDA INSTALLATION
#---------------------------------------------------------------------

# This example of blackbox function is for block evaluation (BB_MAX_BLOCK_SIZE provided)

#x0 = [1,1,1,1]
x0= []
lb = [-1,0,0,0]
ub = [1,5,5,5]
dimPb= len(lb)

def bb(x):
    try:
        nbVal = x.get_n()

        # print("nbVal=",nbVal)

        if ( nbVal % dimPb != 0 ):
            print("Invalid number of values passed to bb")
            return -1

        nbPts = nbVal//dimPb

        print ("----------------- Block eval (",nbPts,") ------------------")
        for i in range(nbPts):
            # Rosenbrock
            f = sum([(100*((x.get_coord(j) ** 2 - x.get_coord(j+1)) ** 2 ) + (x.get_coord(j)-1) ** 2 )  for j in range(i*dimPb,(i+1)*dimPb-1)])
            x.set_bb_output(i, f)
            for j in range(i*dimPb,(i+1)*dimPb):
                print(x.get_coord(j),end=" ")
            print("=> f=",f)

    except:
        print ("Unexpected error in bb()", sys.exc_info()[0])
        return -1
    return 1 



params = ['BB_INPUT_TYPE (I R R R)','BB_OUTPUT_TYPE OBJ','MAX_BB_EVAL 128' ,'BB_MAX_BLOCK_SIZE 8','DISPLAY_STATS BBE OBJ','MODEL_SEARCH SGTELIB','SGTELIB_MODEL_CANDIDATES_NB 8','SGTELIB_MODEL_TRIALS 5','MODEL_EVAL_SORT no','DIRECTION_TYPE ORTHO 2N','SPECULATIVE_SEARCH no','LH_SEARCH 8 0','OPPORTUNISTIC_EVAL false','NM_SEARCH false','PERIODIC_VARIABLE 0'] #,'GRANULARITY (1 0.001 0.001 0.001)']
[ x_return , f_return , h_return, nb_evals , nb_iters ,  stopflag ] = PyNomad.optimize(bb,x0,lb,ub,params)
print ('\n NOMAD outputs \n X_sol={} \n F_sol={} \n H_sol={} \n NB_evals={} \n NB_iters={} \n'.format(x_return,f_return,h_return,nb_evals,nb_iters))
