"""
broad crested weir 2-D
"""
from __future__ import division
from past.utils import old_div
import numpy as np
from proteus import (Domain, Context)
from proteus.Profiling import logEvent
from proteus.mprans.SpatialTools import Tank2D
from proteus.mprans import SpatialTools as st
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
from proteus.Gauges import PointGauges, LineIntegralGauges, LineGauges
from proteus.ctransportCoefficients import smoothedHeaviside



# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts= Context.Options([
    ("final_time",2.0,"Final time for simulation"),
    ("dt_output",0.01,"Time interval to output solution"),
    ("cfl",0.3,"Desired CFL restriction"),
    ("he",0.1,"Maximum element edge length"),
    ("inflow_vel",0.37,"inflow velocity for left boundary"),
    ])


waterLine_y = 0.65
#waterLine_x = 2.5
outflow_level=0.6
water_level=waterLine_y
# Water
rho_0 = 998.2
nu_0 = 1.004e-6

# Air
rho_1 = 1.205
nu_1 = 1.500e-5

g = [0., -9.81, 0.]
# ****************** #
# ***** GAUGES ***** #
# ****************** #

# *************************** #
# ***** DOMAIN AND MESH ***** #
# ****************** #******* #
domain = Domain.PlanarStraightLineGraphDomain()

# ----- TANK ----- #
boundaryOrientations = {'y-': np.array([0., -1.,0.]),
                        'x+': np.array([+1., 0.,0.]),
                        'y+': np.array([0., +1.,0.]),
                        'x-': np.array([-1., 0.,0.]),
#                        'airvent': np.array([-1., 0.,0.]),
                           }
boundaryTags = {'y-' : 1,
                'x+' : 2,
                'y+' : 3,
                'x-' : 4,
#                'airvent':5,
               }

top = 1.0
width = 0.6
length = 5.0
upstream_length = 1.75
vertices=[[0.0, 0.0],#0
          [upstream_length, 0.0],#1
          [upstream_length+0.06, 0.06], #2
          [upstream_length+0.06+1.38, 0.06],#3
          [upstream_length+1.5, 0.0],#4                                                      
          [length, 0.0],#5
          [length, top],#6
          [0.0,top],#7
]


vertexFlags=np.array([4, 
                      1, 1, 1, 1,
                      2, 2,
                      3])

segments=[[0,1],#0
          [1,2],#1
          [2,3],#2
          [3,4],#3
          [4,5],#4 airvent
          [5,6],#5
          [6,7],#6
          [7,0]]#7

segmentFlags=np.array([1, 1, 1, 1, 1,
                       2,
                       3,
                       4])

#####adding refinement to mesh#####
refverts = [[upstream_length*3/4,0.06*3],[length-(upstream_length*3/4),0.06*3],[upstream_length*3/4,0.06*6],[length-(upstream_length*3/4),0.06*6]]
vertices = vertices+refverts

refFlags=np.array([1,1,1,1])
vertexFlags=np.append(vertexFlags,refFlags)


refSegs = [[8,9],[9,11],[11,10],[10,8]]
segments = segments+refSegs

refFlags=[1,1,1,1]
segmentFlags=np.append(segmentFlags,refFlags)


regions = [[2.5, 0.06*4],[2.5, 0.06*2]]
regionFlags =np.array([1,2])
regionConstraints=[0.1,0.02]#[opts.he,0.2*opts.he]

#domain = Domain.PlanarStraightLineGraphDomain(vertices=vertices,
#                                              vertexFlags=vertexFlags,
#                                              segments=segments,
#                                              segmentFlags=segmentFlags,
#                                              regions=regions,
#                                              regionFlags=regionFlags,
#                                              regionConstraints=regionConstraints)
#)


#tank = st.CustomShape(domain,
#                      boundaryTags=boundaryTags, boundaryOrientations=boundaryOrientations)


#regions = [ [ 0.1 , 0.1] ]

#regionFlags=np.array([1])

tank = st.CustomShape(domain, vertices=vertices, vertexFlags=vertexFlags,
                      segments=segments, segmentFlags=segmentFlags,
                      regions=regions, regionFlags=regionFlags,
                      regionConstraints=regionConstraints,
                      boundaryTags=boundaryTags, boundaryOrientations=boundaryOrientations)


# ----- EXTRA BOUNDARY CONDITIONS ----- #
tank.BC['y+'].setAtmosphere()
#tank.BC['y-'].setFreeSlip()
tank.BC['y-'].setNoSlip()
tank.BC['x+'].setHydrostaticPressureOutletWithDepth(seaLevel=outflow_level,
                                                    rhoUp=rho_1,
                                                    rhoDown=rho_0,
                                                    g=g,
                                                    refLevel=top,
                                                    smoothing=1.5*opts.he,
)
#tank.BC['x+'].setNoSlip()

def rampofXT(x,t):
    #if t < 60.0:
    #    return 0.
    if t < 2.0:
        return (t)/2.
    else:
        return 1.0

def phi_inflow(x,t):
    return x[1]-water_level

def vof_inflow(x,t): 
    if phi_inflow(x,t)>0:
        return 1.0
    else:
        return 0.0
    #return smoothedHeaviside(1.5*opts.he,phi_inflow(x,t))

def pressure_inflow(x,t):
    if x[1]<water_level:
        return (top - water_level)*rho_1*9.81 + (water_level - x[1])*rho_0*9.81
    else:
        return (top - x[1])*rho_1*9.81

def velocity_inflow(x,t):
    return (1-vof_inflow(x,t))*opts.inflow_vel

def p_adv(x,t):
    return -1*(1-vof_inflow(x,t))*opts.inflow_vel
    
tank.BC['x-'].reset()
tank.BC['x-'].p_dirichlet.uOfXT = lambda x, t: pressure_inflow(x,t)
#tank.BC['x-'].p_advective.uOfXT = lambda x, t: rampofXT(x,t)*p_adv(x,t)
tank.BC['x-'].u_dirichlet.uOfXT = lambda x,t: rampofXT(x,t)*velocity_inflow(x,t)  
tank.BC['x-'].v_dirichlet.uOfXT = lambda x, t: 0.0
tank.BC['x-'].phi_dirichlet.uOfXT = lambda x, t: phi_inflow(x,t)
tank.BC['x-'].vof_dirichlet.uOfXT = lambda x, t: vof_inflow(x,t)
tank.BC['x-'].u_diffusive.uOfXT = lambda x, t: 0.0
tank.BC['x-'].v_diffusive.uOfXT = lambda x, t: 0.0

domain.MeshOptions.he = opts.he
st.assembleDomain(domain)
domain.MeshOptions.triangleOptions = "VApq30Dena%8.8f" % ((opts.he ** 2)/2.0,)

# ****************************** #
# ***** INITIAL CONDITIONS ***** #
# ****************************** #
class zero(object):
    def uOfXT(self,x,t):
        return 0.0

class PHI_IC:
    def uOfXT(self, x, t):
#        phi_x = x[0] - waterLine_x
        phi_y = x[1] - waterLine_y
        phi_y_outflow = x[1] - outflow_level
#        if phi_x <= 0.0:
#            if phi_y < 0.0:
#                return max(phi_x, phi_y)
        return phi_y
#            else:
#                return phi_y
#        else:
#            if phi_y_outflow < 0.0:
#                return phi_y_outflow
#            else:
#                if phi_y<0.0:
#                    return min(phi_x, phi_y_outflow)
#                else:
#                    return min((phi_x ** 2 + phi_y ** 2)**0.5, phi_y_outflow)

class VF_IC:
    def __init__(self):
        self.phi=PHI_IC()
    def uOfXT(self, x, t):
        from proteus.ctransportCoefficients import smoothedHeaviside
        return smoothedHeaviside(1.5*opts.he,self.phi.uOfXT(x,t))

############################################
# ***** Create myTwoPhaseFlowProblem ***** #
############################################
outputStepping = TpFlow.OutputStepping(opts.final_time,dt_output=opts.dt_output)
initialConditions = {'pressure': zero(),
                     'pressure_increment': zero(),
                     'vel_u': zero(),
                     'vel_v': zero(),
                     'vof': VF_IC(),
                     'ncls': PHI_IC(),
                     'rdls': PHI_IC()}
#                     'clsvof': clsvof_init_cond()}

myTpFlowProblem = TpFlow.TwoPhaseFlowProblem(ns_model=0,
                                             ls_model=0,
                                             nd=2,
                                             cfl=opts.cfl,
                                             outputStepping=outputStepping,
                                             he=opts.he,
                                             domain=domain,
                                             initialConditions=initialConditions)
# copts=myTpFlowProblem.Parameters.Models.rans2p.p.CoefficientsOptions
# def getPhiDBC(x, flag):
#     if flag == boundaryTags['x-']:
#         return lambda x,t: x[1] - waterLine_y
#     elif flag == boundaryTags['x+']:
#         return lambda x,t: x[1] - outflow_level
# myTpFlowProblem.Parameters.Models.ncls.p.dirichletConditions = {0: getPhiDBC}

