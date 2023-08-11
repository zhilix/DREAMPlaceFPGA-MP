##
# @file   Placer.py
# @author Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
# @date   Sep 2020
# @brief  Main file to run the entire placement flow. 
#

import matplotlib 
matplotlib.use('Agg')
import os
import sys 
import time 
import numpy as np 
import logging
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
	sys.path.append(root_dir)
import dreamplacefpga.configure as configure 
from Params import *
from PlaceDB import *
from NonLinearPlace import *
from IFWriter import * 
import pdb 

def placeFPGA(params):
    """
    @brief Top API to run the entire placement flow. 
    @param params parameters 
    """
    assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"

    np.random.seed(params.random_seed)
    # Read Database
    start = time.time()
    placedb = PlaceDBFPGA()
    placedb(params) #Call function
    #logging.info("Reading database takes %.2f seconds" % (time.time()-start))

    # write out xdc file
    # placedb.writeXDC(params, "design_constr.xdc")

    # Random Initial Placement 
    placer = NonLinearPlaceFPGA(params, placedb)
    #logging.info("non-linear placement initialization takes %.2f seconds" % (time.time()-tt))
    metrics = placer(params, placedb)
    logging.info("Placement completed in %.2f seconds" % (time.time()-start))

    # write placement solution 
    path = "%s/%s" % (params.result_dir, params.design_name())
    
    logging.info("Completed Placement in %.3f seconds" % (time.time()-start))

    # write macro placement solution
    logging.info("Writing macro placement solution")
    placedb.writeMacroPl(params, "macroplacement.pl")
               

if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow. 
    """
    logging.root.name = 'DREAMPlaceFPGA'
    logging.basicConfig(level=logging.INFO, format='[%(levelname)-7s] %(name)s - %(message)s', stream=sys.stdout)

    if len(sys.argv) < 2:
        logging.error("Input parameters required in json format")
    paramsArray = []
    for i in range(1, len(sys.argv)):
        params = ParamsFPGA()
        params.load(sys.argv[i])
        paramsArray.append(params)
    logging.info("Parameters[%d] = %s" % (len(paramsArray), paramsArray))

    #Settings to minimze non-determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False 
    torch.manual_seed(params.random_seed)
    np.random.seed(params.random_seed)
    #random.seed(params.random_seed)
    if params.gpu:
        torch.cuda.manual_seed_all(params.random_seed)
        torch.cuda.manual_seed(params.random_seed)

    # tt = time.time()
    for params in paramsArray: 
        placeFPGA(params)
    # logging.info("Completed Placement in %.3f seconds" % (time.time()-tt))

