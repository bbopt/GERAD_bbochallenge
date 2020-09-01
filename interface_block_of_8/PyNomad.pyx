# C.Tribes sept. 6th, 2016 --- PyNomad: Beta integration of Nomad in Python 
# PyNomad 1.0

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.list cimport list
from libcpp cimport bool

from cython.operator cimport dereference as deref, preincrement as inc

def version():
    printPyNomadVersion()
 
# Define the interface function to display nomad general information
def usage():
    printPyNomadUsage()
 
# Define the interface function to display nomad general information
def info():
    printPyNomadUsage()
    printPyNomadInfo()

# Define the interface function to get nomad help
def help(about=''):
    about = about.encode(u"ascii")
    printNomadHelp(about)
    
def __doc__():
    cdef string about;
    printPyNomadUsage()
    help(about)	  

# Define the interface function to perform optimization          
def optimize(f , pX0, pLB,pUB ,params):
    cdef PyNomadEval_Point_Ptr u_feas = PyNomadEval_Point_Ptr()
    cdef PyNomadEval_Point_Ptr u_infeas = PyNomadEval_Point_Ptr()
    cdef int run_status
    cdef int nb_evals = 0
    cdef int nb_iters = 0
    cdef double f_return
    cdef double h_return
    x_return=[]
    cdef int nb_params = len(params)
    for i in range(nb_params):
         params[i] = params[i].encode(u"ascii")
    run_status = runNomad(cb, cbL, <void*> f, <vector[double]&> pX0, <vector[double]&> pLB, <vector[double]&> pUB , <vector[string]&> params , u_feas.c_ep ,u_infeas.c_ep , nb_evals, nb_iters)
    if u_feas.c_ep != NULL:
         f_return = u_feas.get_f()
         h_return = 0
         for i in xrange(u_feas.get_n()):
             x_return.append(u_feas.get_coord(i))
         del u_feas.c_ep  
    if u_infeas.c_ep != NULL:
         f_return = u_infeas.get_f()
         h_return = u_infeas.get_h()
         for i in xrange(u_infeas.get_n()):
             x_return.append(u_infeas.get_coord(i))
         del u_infeas.c_ep      
    return [ x_return , f_return, h_return, nb_evals , nb_iters  , run_status ]
   
cdef extern from "Double.hpp" namespace "NOMAD":
    cdef cppclass Double:
        const double & value()
        bool is_defined()
    
cdef class PyNomadDouble:
    cdef Double c_d 
    def value(self):
        return self.c_d.value()
    def is_defined(self):
        return self.c_d.is_defined()

cdef extern from "Point.hpp" namespace "NOMAD":
    cdef cppclass Point:
        const Double & get_coord(int i)
        void set_coord(int i, double &v)
        bool is_defined()

cdef class PyNomadPoint:
    cdef Point c_p
    def get_coord(self,int i):
        cdef PyNomadDouble v = PyNomadDouble()
        v.c_d=self.c_p.get_coord(i)
        cdef double v_d
        if ( v.is_defined() ):
            v_d = v.value()
        else:
            v_d = float('inf')
        return v_d
    def set_coord(self,int i, double v):
        self.c_p.set_coord(i, v)
    def is_defined(self):
        return self.c_p.is_defined()
	       
cdef extern from "Eval_Point.hpp" namespace "NOMAD":
    cdef cppclass Eval_Point:
        const double & value(int i)
        void set_coord(int i, double &v)
        const Double & get_f()
        const Double & get_h()
        void set_bb_output(int i, double & v)
        const Point & get_bb_outputs()
        int get_n()
        int get_m()
        void set(int n, int m)
                     
cdef class PyNomadEval_Point_Ptr:
    cdef Eval_Point *c_ep    # A pointer to an eval point residing in Nomad
    def get_coord(self, int i):       
        return self.c_ep.value(i)
    def get_bb_output(self,int i):
        cdef PyNomadPoint out_p = PyNomadPoint()
        out_p.c_p = self.c_ep.get_bb_outputs()
        if ( out_p.is_defined() ):
            return out_p.get_coord(i)
    def set_bb_output(self,int i, double v):
        self.c_ep.set_bb_output(i,v)
    def get_f(self):
        cdef PyNomadDouble f = PyNomadDouble()
        f.c_d=self.c_ep.get_f()
        cdef double f_d
        if ( f.is_defined() ): 
            f_d = f.value()
        else:
            f_d = float('inf')
        return f_d  
    def get_h(self):
        cdef PyNomadDouble h = PyNomadDouble()
        h.c_d=self.c_ep.get_h()
        cdef double h_d
        if ( h.is_defined() ):
            h_d = h.value()
        else:
            h_d = 0
        return h_d  

    def get_n(self):
        cdef int n
        n = self.c_ep.get_n()
        return n
    def get_m(self):
        cdef int m
        m = self.c_ep.get_m()
        return m
#    def display_eval(self):
#        printEvalPoint(self.c_ep)

cdef class PyNomadEval_Point:
    cdef Eval_Point ep     # A new eval point to be given to python blackbox
    def get_coord(self, int i):
        return self.ep.value(i)
    def set_coord(self, int i, double v):
        self.ep.set_coord(i,v)
    def get_bb_output(self,int i):
        cdef PyNomadPoint out_p = PyNomadPoint()
        out_p.c_p = self.ep.get_bb_outputs()
        if ( out_p.is_defined() ):
            return out_p.get_coord(i)
    def set_bb_output(self,int i, double v):
        self.ep.set_bb_output(i,v)
    def get_f(self):
        cdef PyNomadDouble f = PyNomadDouble()
        f.c_d=self.ep.get_f()
        cdef double f_d
        if ( f.is_defined() ):
            f_d = f.value()
        else:
            f_d = float('inf')
        return f_d
    def get_h(self):
        cdef PyNomadDouble h = PyNomadDouble()
        h.c_d=self.ep.get_h()
        cdef double h_d
        if ( h.is_defined() ):
            h_d = h.value()
        else:
            h_d = 0
        return h_d

    def get_n(self):
        cdef int n
        n = self.ep.get_n()
        return n
    def get_m(self):
        cdef int m
        m = self.ep.get_m()
        return m
    def set(self, int n, int m):
        self.ep.set(n,m)
    def display(self):
        printEvalPoint(self.ep)

cdef extern from "nomadCySimpleInterface.cpp":
    ctypedef int (*Callback)(void * apply, Eval_Point& x,bool hasSgte , bool sgte_eval)
    ctypedef int (*CallbackL)(void * apply, list[Eval_Point *] & x,bool hasSgte , bool sgte_eval)
    void printPyNomadInfo()
    void printEvalPoint(const Eval_Point &)
    void printPyNomadUsage()
    void printNomadHelp( string about)
    void printPyNomadVersion()
    int runNomad(Callback cb, CallbackL cbL, void* apply, vector[double] &X0, vector[double] &LB, vector[double] &UB , vector[string] & params , Eval_Point *& best_feas_sol ,Eval_Point *& best_infeas_sol , int & nb_evals, int & nb_iters) except+

# Define callback function for a single Eval_Point ---> link with Python     
cdef int cb(void *f, Eval_Point & x, bool hasSgte , bool sgte_eval ):
     cdef PyNomadEval_Point_Ptr u = PyNomadEval_Point_Ptr()
     u.c_ep = &x
     if ( hasSgte ):
        return (<object>f)(u,sgte_eval)
     else:
        return (<object>f)(u)
       
# Define callback function for block evaluation of a list of Eval_Points ---> link with Python
cdef int cbL(void *f, list[Eval_Point *] & x, bool hasSgte , bool sgte_eval ):
      cdef size_t size = x.size()
      cdef PyNomadEval_Point u = PyNomadEval_Point()   # Create a new (big) eval point
      cdef list[Eval_Point *].iterator it = x.begin()
      cdef Eval_Point *c_ep

      #
      # Put all the Eval_Point of the list into a single eval point u
      #
      cdef int posG
      posG=0

      # Dimension
      cdef size_t n
      c_ep = deref(it)
      # print("value=",c_ep.value(0))
      n =  c_ep.get_n()

      # Number of output
      cdef size_t m
      m=c_ep.get_m()

      # print("nbBBO=",nbBBO,"dim=",dim," size=",size)

      # Set the dimension of a (big) eval point containing all eval points from nomad
      u.set(n*size,m*size)

      # print("dim u=",u.get_n()," nBBO u =",u.get_m())

      for i in xrange(size):
         c_ep = deref(it)
         for j in range(n):
            # print("posG=",posG," value=",c_ep.value(j))
            u.set_coord(posG, c_ep.value(j))
            posG=posG+1

         inc(it)

      if ( hasSgte ):
         (<object>f)(u,sgte_eval)
      else:
         (<object>f)(u)



      #
      # Update the Eval_Points bb_output
      #
      it = x.begin()
      c_ep = deref(it)

      cdef double bbo
      for i in xrange(size):
          c_ep = deref(it)
          for j in range(m):
             bbo = u.get_bb_output(i*m+j)
             # print("n_bbo=",i*m+j,"bbo=",bbo)
             c_ep.set_bb_output(j,bbo)

          inc(it)

      return 1   # 1 is success
