import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
import logging
root = logging.root
for h in root.handlers:
	h.setLevel("INFO")
logger = logging.getLogger(__name__)

def get_parameters(fn):
	values = []
	with open(fn) as f:
		parameters = f.readlines()
	
	for p in parameters:
		values.append(float(p[:p.index(' ')]))
	
	return values
		
folder = 'experiment2/'
####################################################
# SHAPE
####################################################

from scipy.special import erf
from numpy import exp

def mask(x, z, phi, delta, lx, lz, h, t, xpos, zpos):
    cos, sin = np.cos(phi), np.sin(phi)
    x1, x2 = (x * cos + z * sin), (z * cos - x * sin)
    return 0.5 * (1 - erf(C(x1, x2, lx, lz, h, t, xpos, zpos) / delta))

def C(x, z, lx, lz, h, t, xpos, zpos):
    Z0 = -1
    m, n = 2, 2
    Z = np.abs((x - xpos) / (lx + h * exp(-(z-zpos)**2 / t**2))) ** m + np.abs((z - zpos) / lz) ** n + Z0
    return Z

####################################################
# DOMAIN
####################################################
buffer = .5
Lx, Ly = 5, 4.5
dx, dz = 5e-3, 5e-3

nx, ny = 2 * Lx / dx, 2 * Ly / dz

xbasis = de.Fourier('x',int(nx),interval=(-Lx-buffer,Lx+buffer),dealias=3/2)
ybasis = de.Fourier('y',int(ny),interval=(-Ly-buffer,Ly+buffer),dealias=3/2)
domain = de.Domain([xbasis,ybasis],grid_dtype=np.float64)

x,y = domain.grids(domain.dealias)
xx, yy = x+0*y, 0*x+y

####################################################
# Parameters
####################################################
[ν, μ, γ, δ, φ_deg, lx, lz, h, t, xpos, zpos, b_speed] = get_parameters(f'{folder}parameters.txt')

φ = φ_deg * np.pi / 180

# Source speed
b_start = -Lx + 0.5

timestepper = 'SBDF2'

####################################################
# BUILD WALL
####################################################

wall = domain.new_field()
wall.set_scales(domain.dealias)

wall['g'] = 1 + 0.5*(np.tanh((x-Lx)/.05) + np.tanh(-(x+Lx)/.05))

####################################################
# BUILD SOURCES
####################################################

n_sources = 9
yss = np.linspace(-Ly,Ly,9)

sources = domain.new_field()
sources.set_scales(domain.dealias)

for yi in yss:
    r = np.hypot((xx-b_start),(yy-yi))
    sources['g'] += np.exp(-(r/0.01)**2)

####################################################
# BUILD SHAPE
####################################################

Γ = domain.new_field()
Γ.set_scales(domain.dealias)
Γ['g'] = mask(x, y, φ, δ, lx, lz, h, t, xpos, yss[int(0.5 * (n_sources - 1))])

####################################################
# DEFINE PROBLEM
####################################################

problem = de.IVP(domain,variables=['ux','uy','p','b'])
problem.parameters['ν'] = ν
problem.parameters['μ'] = μ
problem.parameters['γ'] = γ
problem.parameters['δ'] = δ
problem.parameters['Γ'] = Γ
problem.parameters['wall'] = wall
problem.parameters['sources'] = sources
problem.parameters['b_speed'] = b_speed
problem.substitutions['inflow_speed'] = '0.5 * b_speed * (1 + tanh((t - 0.15) / 0.1))'
problem.substitutions['q'] = 'dx(uy) - dy(ux)'
problem.substitutions['bx'] = 'dx(b)'
problem.substitutions['by'] = 'dy(b)'


problem.add_equation('dx(ux) + dy(uy) = 0',condition='(nx != 0) or (ny != 0)')
problem.add_equation('p = 0',condition='(nx == 0) and (ny == 0)')
problem.add_equation('dt(ux) + dx(p) + ν*dy(q) =  q*uy - γ*Γ*ux - γ*wall*(ux-inflow_speed)')
problem.add_equation('dt(uy) + dy(p) - ν*dx(q) = -q*ux - γ*Γ*uy')
problem.add_equation('dt(b) - μ*(dx(bx) + dy(by)) = -(ux*bx + uy*by) - γ*wall*b - γ*sources*(b-inflow_speed)')

####################################################
# BUILD SOLVER
####################################################

solver = problem.build_solver(eval(f'de.timesteppers.{timestepper}'))
solver.stop_sim_time = 20

ux,uy,p,b = [solver.state[a] for a in problem.variables]

for field in ux,uy,p,b:
    field['g'] = 0
   
####################################################
# BUILD SNAPSHOTS
####################################################

snapshots = solver.evaluator.add_file_handler(f'{folder}data', iter=100)
for task in ['ux', 'uy', 'q', 'b', 'Γ']:
	snapshots.add_task(task)

q = solver.evaluator.vars['q']


####################################################
# DEFINE FORCES
####################################################

force = flow_tools.GlobalFlowProperty(solver, cadence=1)
force.add_property("Γ*ux", name='fx', precompute_integral=True)
force.add_property("Γ*uy", name='fy', precompute_integral=True)

####################################################
# ITERATION
####################################################

while solver.ok:
    solver.step(5e-4)
    if solver.iteration % 10 == 0:
        if solver.iteration % 50 == 0:
            logger.info("Ax: {:.2e}".format(force.properties['_fx_integral']['g'][0,0]))
            logger.info("Ay: {:.2e}".format(force.properties['_fy_integral']['g'][0,0]))               
            
