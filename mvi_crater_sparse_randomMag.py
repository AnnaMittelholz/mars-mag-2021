#!/usr/bin/env python
# coding: utf-8
# general python ecosystem

# %%
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import ipywidgets
import random
import math

# SimPEG tools 
import discretize
from discretize.utils import mesh_builder_xyz, refine_tree_xyz
from SimPEG.potential_fields import magnetics as mag
from SimPEG.utils import mkvc, surface2ind_topo
from SimPEG import (
    data,
    data_misfit,
    directives,
    maps,
    inverse_problem,
    optimization,
    inversion,
    regularization,
    utils
)
#from pymatsolver import Pardiso as Solver

# In[3]: Flags for inflight sampling and large scale noise 
    
inflight=2     # inflight sampling is 1; landed sampling is 0; 2 is mesh
gauss_noise=1  # large wavelength noise on is 1.
use_topo=True 
     
    # %%

target_magnetization_inclination = 45
target_magnetization_declination = 90 

target_magnetization_direction = utils.mat_utils.dip_azimuth2cartesian(
    target_magnetization_inclination, target_magnetization_declination
)

target_magnetization_amplitude = 0 # magnetization in A/m
background_magnetization = 10 # magnetization in A/m
target_magnetization = target_magnetization_amplitude * target_magnetization_direction

# In[4]: TOPO
# Create grid of points for topography
# Use a Gaussian topo to simulate crater
[xx, yy] = np.meshgrid(np.linspace(-600, 600, 50), np.linspace(-600, 600, 50))
b = 100
A = 50

if use_topo is True:
    zz = -A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
else: 
    zz = np.zeros_like(xx)
    
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# %%obs pints 
if inflight ==1:  #inflight measuremenets 
    line_length = 600
    n_data_along_line = 30
    survey_x = np.linspace(-line_length/2, line_length/2, n_data_along_line)
    survey_y = np.r_[-50,0, 50] 
    survey_z = np.r_[10]
    survey_xyz = discretize.utils.ndgrid([survey_x, survey_y, survey_z])
    
    xx = np.linspace(0, 7, n_data_along_line)
    xx =np.concatenate([xx,xx,np.flip(xx)])
    
    survey_xyz[:,1] = survey_xyz[:,1]*xx
    survey_z = np.linspace(1, 30, int(n_data_along_line/2))
    zz = np.concatenate([survey_z, np.flip(survey_z), survey_z, np.flip(survey_z), survey_z, np.flip(survey_z)])
    survey_xyz[:,2] = zz

elif inflight == 2: # grid
    line_length = 600
    n_data_along_line = 15
    survey_x = np.linspace(-line_length/2, line_length/2, n_data_along_line)
    survey_y = survey_x
    survey_z = np.r_[10]
    survey_xyz = discretize.utils.ndgrid([survey_x, survey_y, survey_z])

else:
    line_length = 600
    n_data_along_line = 6
    survey_x = np.linspace(-line_length/2, line_length/2, n_data_along_line)
    survey_y = np.r_[0] 
    survey_z = np.r_[1]
    survey_xyz = discretize.utils.ndgrid([survey_x, survey_y, survey_z])
  

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.scatter3D(survey_xyz[:, 0], survey_xyz[:, 1], survey_xyz[:, 2], c="k", alpha=1)
plt.show()
      
# %%  PLOT 
# Here how the topography looks with a quick interpolation, just a Gaussian...
tri = sp.spatial.Delaunay(topo)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.plot_trisurf(
    topo[:, 0], topo[:, 1], topo[:, 2], triangles=tri.simplices, cmap=plt.cm.Spectral
)
ax.scatter3D(survey_xyz[:, 0], survey_xyz[:, 1], survey_xyz[:, 2], c="k", alpha=1)
plt.show()

# %% make mesh

from discretize import TensorMesh

nc = 40  # number of core mesh cells in x, y and z
dh = 20   # base cell width in x, y and z
npad = 10  # number of padding cells
exp = 1  # expansion rate of padding cells

h = [(dh, npad, -exp), (dh, nc), (dh, npad, exp)]
mesh = TensorMesh([h, h, h], x0="CCC")

# Define an active cells from topo
actv = utils.surface2ind_topo(mesh, topo)
nC = int(actv.sum())
model_map = maps.IdentityMap(nP=nC)  # model is a vlue for each active cell

mesh

# %%  make Geometry of magnetized body  

ind = utils.model_builder.getIndicesSphere([0,0,-40], 120, mesh.gridCC)

magnetization = np.ones((mesh.nC, 3))*background_magnetization
magnetization[ind, :] = target_magnetization
model = magnetization[actv, :]


if gauss_noise > 0:  #inflight measuremenets 

    mu, sigma = 2, 2 # mean and standard deviation
    nums = [] 

    for i in range(3*mesh.nC): 
        magnetization = random.gauss(mu, sigma)
        nums.append(magnetization) 

    magnetization = np.reshape(nums, (mesh.nC,3))

    mag_f = sp.ndimage.gaussian_filter(magnetization,sigma=4,order=0, output=None, mode='nearest', cval=0.0, truncate=4.0)
    mag_f[ind, :] = target_magnetization
    model_f = mag_f[actv, :]


active_cell_map = maps.InjectActiveCells(mesh=mesh, indActive=actv, valInactive=np.nan)

# %%

def full_mesh_magnetization(model):
    return np.vstack([active_cell_map * model.reshape(nC, 3, order="F")[:, i] for i in range(3)]).T


def plot_vector_model(
    maxval, model, ax=None, quiver_opts=None, normal="Y", xlim=None, ylim=None, ind=None, plot_data=True, plot_grid=False
):
    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    magnetization = full_mesh_magnetization(model)
    qo = {
        "units":"xy", "scale":np.max(np.abs(model))/20,
        "headwidth":7, "headlength":10, "headaxislength":10
    }
    
    # overwrite default vals if user provides them 
    if quiver_opts is not None:
        for key, val in quiver_opts.items(): 
            qo[key] = val 

    cb = plt.colorbar(
        mesh.plot_slice(
            magnetization, "CCv", clim=[0,maxval],normal=normal, ax=ax, view="vec", 
            grid=plot_grid, ind=ind, quiver_opts=qo
        )[0], ax=ax
    )
    
    cb.set_label("amplitude magnetization (A/m)")

    if normal.upper() == "X": 
        if plot_data is True: 
            ax.plot(survey_xyz[:, 1], survey_xyz[:, 2], "C1o", ms=2,label=False)
        ax.set_xlim([survey_x.min()*1.5, survey_x.max()*1.5] if xlim is None else xlim)
      #  ax.set_ylim([target_geometry[:,2].min()*3, survey_z.max()*4] if ylim is None else ylim)
    elif normal.upper() == "Y": 
        if plot_data is True: 
            ax.plot(survey_xyz[:, 0], survey_xyz[:, 2], "C1o", ms=2)
        ax.set_xlim([survey_x.min()*1.5, survey_x.max()*1.5] if xlim is None else xlim)
      #  ax.set_ylim([target_geometry[:,2].min()*3, survey_z.max()*4] if ylim is None else ylim)
    elif normal.upper() == "Z": 
        if plot_data is True: 
            ax.plot(survey_xyz[:, 0], survey_xyz[:, 1], "C1o", ms=2)
        ax.set_xlim([survey_x.min()*1.25, survey_x.max()*1.25] if xlim is None else xlim)
        ax.set_ylim([survey_x.min()*1.25, survey_x.max()*1.25] if ylim is None else ylim)
    ax.legend()
    ax.set_aspect(1)
    
def plot_amplitude(
        maxval, model, ax=None, quiver_opts=None, normal="Y", xlim=None, ylim=None, ind=None, plot_data=True, plot_grid=False
    ):
        if ax is None: 
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        qo = {
            "units":"xy", "scale":np.max(np.abs(model))/20,
            "headwidth":7, "headlength":10, "headaxislength":10
        }
        
        # overwrite default vals if user provides them 
        if quiver_opts is not None:
            for key, val in quiver_opts.items(): 
                qo[key] = val 
                
        magnetization = full_mesh_magnetization(model)
        Mtotal = np.sqrt(magnetization[:,0]**2 + magnetization[:,1]**2 + magnetization[:,2]**2)
                
        cb = plt.colorbar(
            mesh.plot_slice(
                Mtotal, "CC", clim=[0,maxval],normal=normal, ax=ax, view="abs", 
                grid=plot_grid, ind=ind, quiver_opts=qo
                )[0], ax=ax
            )
        
        cb.set_label("amplitude magnetization (A/m)")

        if normal.upper() == "X": 
            if plot_data is True: 
                ax.plot(survey_xyz[:, 1], survey_xyz[:, 2], "C1o", ms=2,label=False)
            ax.set_xlim([survey_x.min()*1.5, survey_x.max()*1.5] if xlim is None else xlim)
          #  ax.set_ylim([target_geometry[:,2].min()*3, survey_z.max()*4] if ylim is None else ylim)
        elif normal.upper() == "Y": 
            if plot_data is True: 
                ax.plot(survey_xyz[:, 0], survey_xyz[:, 2], "C1o", ms=2)
            ax.set_xlim([survey_x.min()*1.5, survey_x.max()*1.5] if xlim is None else xlim)
          #  ax.set_ylim([target_geometry[:,2].min()*3, survey_z.max()*4] if ylim is None else ylim)
        elif normal.upper() == "Z": 
            if plot_data is True: 
                ax.plot(survey_xyz[:, 0], survey_xyz[:, 1], "C1o", ms=2)
            ax.set_xlim([survey_x.min()*1.25, survey_x.max()*1.25] if xlim is None else xlim)
            ax.set_ylim([survey_x.min()*1.25, survey_x.max()*1.25] if ylim is None else ylim)
        ax.legend()
        ax.set_aspect(1)
    
# %%  Plot Model
maxval=5

if gauss_noise > 0: 
    
    fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})

    zind = 23
    
    plot_vector_model(maxval,model_f, ax=ax[0])
    plot_vector_model(maxval,model_f, ax=ax[1], normal="Z", ind=zind)  # APPEND WITHOUT GRID,plot_grid=False

    plt.tight_layout()

    fn = 'Model_Gauss.png'

# %%
fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})


zind = 25
plot_vector_model(maxval,model, ax=ax[0])
plot_vector_model(maxval,model, ax=ax[1], normal="Z", ind=zind)  # APPEND WITHOUT GRID,plot_grid=False
plt.tight_layout()

fn = 'Model2.png'


fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})

plot_amplitude(maxval,model, ax=ax[0],plot_grid=True)
plot_amplitude(maxval,model, ax=ax[1], normal="Z", ind=zind,plot_grid=True)  # APPEND WITHOUT GRID,plot_grid=False
plt.tight_layout()

fn = 'Model2.png'

if gauss_noise > 0: 
    model = model_f


# %%

components = ["x", "y", "z"]
rx = mag.receivers.Point(locations=survey_xyz, components=[f"b{comp}" for comp in components])

source_field = mag.sources.SourceField(
    receiver_list=[rx], parameters=np.r_[1., 0, 0]
)
survey = mag.survey.Survey(source_field)

simulation = mag.simulation.Simulation3DIntegral(
    mesh=mesh, survey=survey, chiMap=maps.IdentityMap(nP=np.prod(model.shape)), 
    actInd=actv, model_type="vector"
)

# In[44]: Data for inversion 

synthetic_data = simulation.make_synthetic_data(
    utils.mkvc(model), 
    noise_floor=0.1, # standard deviation of the noise in nT 
    relative_error=0,  # percent noise 
    add_noise=True  # do we add noise to the data we will use in the inversion?
) 

# In[45]:
survey_z_s = [1]

def plot_data_profile(data, plot_opts=None, ax=None, xlim=None, ylim=None, label=True):
    data = data.reshape((survey_xyz.shape[0], len(components)))
    
    if ax is None: 
        fig, ax = plt.subplots(1, len(components), figsize=(5*len(components), 4))
        ax = np.atleast_2d(ax)
    
    po = {"ms": 3}
    if plot_opts is not None: 
        for key, val in plot_opts.items():
            po[key] = val 
    
    for k, zloc in enumerate(survey_z_s): 
        for i, component in enumerate(components):
            d = data[:, i].reshape(len(survey_x), len(survey_y), len(survey_z_s), order="F")
            for j, y in enumerate(survey_y):
                if not isinstance(label, bool):
                    l=f"{y:1.0f} m {label}"
                else:
                    l=f"{y:1.0f} m" if label is True else None
                ax[k, i].plot(survey_x, d[:, j, k], f"C{j}", label=l, **po)

            #ax[k, i].set_title(f"B{component} z={zloc}m")
            ax[k, i].grid("both", alpha=0.6)
            ax[k, i].set_ylim(1.25 * np.r_[data.min(), data.max()] if ylim is None else ylim)
            ax[k, i].set_xlim(xlim)
            ax[k, i].set_xlabel("x (m)")
    
    ax[0, 0].set_ylabel("magnetic field (nT)")
    if label is not False: 
        ax[0, 0].legend()
    plt.tight_layout()
    return ax

# In[46]:

ax=plot_data_profile(synthetic_data.dclean, label=False)
ax=plot_data_profile(synthetic_data.dobs, ax=ax, plot_opts={"marker":"o", "alpha":0.5}, label=False)
 
fn = 'Data_Profiles.png'

# %%
fig, ax = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1,1,1, 1]})

d_obs = synthetic_data.dobs.reshape((survey_xyz.shape[0], len(components)))
B_obs = np.sqrt(d_obs[:,0]**2 + d_obs[:,1]**2 + d_obs[:,2]**2)

vvmin = np.min(d_obs)
vvmax = np.max(d_obs)

if abs(vvmin)<abs(vvmax):
    vvmin = -vvmax
    
if abs(vvmax)<abs(vvmin):
    vvmax = -vvmin
    

ax[0].scatter(survey_xyz[:, 0], survey_xyz[:, 1], marker='o',c=B_obs,s=100, cmap='viridis',vmin=vvmin,vmax=vvmax)
ax[0].set_title("|B|")
ax[1].scatter(survey_xyz[:, 0], survey_xyz[:, 1], marker='o',c=d_obs[:,0],s=100, cmap='viridis',vmin=vvmin,vmax=vvmax)
ax[1].set_title("Bx")
ax[2].scatter(survey_xyz[:, 0], survey_xyz[:, 1], marker='o',c=d_obs[:,1],s=100, cmap='viridis',vmin=vvmin,vmax=vvmax)
ax[2].set_title("By")
sc =ax[3].scatter(survey_xyz[:, 0], survey_xyz[:, 1], marker='o',c=d_obs[:,2],s=100, cmap='viridis',vmin=vvmin,vmax=vvmax)
ax[3].set_title("Bz")
cbar = fig.colorbar(sc)


fn = 'Data_Profiles.png'

# %% NOW KEEP GEOMETRY


# %%
# create the regularization
wires = maps.Wires(("x", nC), ("y", nC), ("z", nC))

reg_x = regularization.Sparse(mesh, indActive=actv, mapping=wires.x)#, alpha_s=1e-6)
reg_y = regularization.Sparse(mesh, indActive=actv, mapping=wires.y)#, alpha_s=1e-6)
reg_z = regularization.Sparse(mesh, indActive=actv, mapping=wires.z)#, alpha_s=1e-6)

norms = [[2, 2, 2, 2]]
reg_x.norms = norms
reg_y.norms = norms
reg_z.norms = norms

reg_x.objfcts = reg_x.objfcts[:-1]
reg_y.objfcts = reg_y.objfcts[:-1]
reg_z.objfcts = reg_z.objfcts[:-1]

reg = reg_x + reg_y + reg_z

# simulation2.G dimensions mismatch 
dmis = data_misfit.L2DataMisfit(data=synthetic_data, simulation=simulation)

# optimization
opt = optimization.InexactGaussNewton(
    maxIter=20, maxIterLS=20, maxIterCG=20, tolCG=1e-4
)

# inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# directives 
betaest = directives.BetaEstimate_ByEig(beta0_ratio=6)  # estimate initial trade-off parameter
sensitivity_weights = directives.UpdateSensitivityWeights()  # Add sensitivity weights
IRLS = directives.Update_IRLS()  # IRLS
update_Jacobi = directives.UpdatePreconditioner()  # Pre-conditioner
target = directives.TargetMisfit(chifact=1)  # target misfit 


inv = inversion.BaseInversion(
    inv_prob, directiveList=[sensitivity_weights, IRLS, update_Jacobi, betaest, target]
)

 # In[57]:run the inversion

m0 = np.zeros(nC * 3)
mrec_cartesian = inv.run(m0)

# In[57]:


fig, ax = plt.subplots(2, 2, figsize=(17, 10), gridspec_kw={'width_ratios': [2, 1]})

quiver_opts = {
    "scale":np.max(np.abs(mrec_cartesian))/20,
}

plot_vector_model(maxval,mrec_cartesian, ax=ax[0,0])
plot_vector_model(maxval,mrec_cartesian, ax=ax[0,1], normal="Z", ind=25)
ax[0,0].set_title(f"y={mesh.vectorCCy[30]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[22]}")
quiver_opts='None'
plot_amplitude(maxval,mrec_cartesian, ax=ax[1,0])
plot_amplitude(maxval,mrec_cartesian, ax=ax[1,1], normal="Z", ind=25)
ax[1,0].set_title(f"")
ax[1,1].set_title(f"")
plt.tight_layout()
fn = 'L2_invmodel.png'
#plt.savefig(pn+fn)

# In[58]:


ax = plot_data_profile(synthetic_data.dobs, plot_opts={"marker":"o", "lw":0}, label=False)#"observed")
ax = plot_data_profile(inv_prob.dpred, ax=ax, label=False)#"predicted")
 
fn = 'L2_inv_obs_pred.png'
#plt.savefig(pn+fn)

# %%
fig,ax = plt.subplots(3,3,figsize=(15, 10))
quiver_opts = {
    "scale":np.max(np.abs(mrec_cartesian))/20,
}

plot_vector_model(maxval,mrec_cartesian, ax=ax[0,0], ind=30)
plot_vector_model(maxval,mrec_cartesian, ax=ax[0,1], normal="Z", ind=28)
plot_vector_model(maxval,mrec_cartesian, ax=ax[0,2], normal="Z", ind=27)
plot_vector_model(maxval,mrec_cartesian, ax=ax[1,0], normal="Z", ind=26)
plot_vector_model(maxval,mrec_cartesian, ax=ax[1,1], normal="Z", ind=25)
plot_vector_model(maxval,mrec_cartesian, ax=ax[1,2], normal="Z", ind=24)
plot_vector_model(maxval,mrec_cartesian, ax=ax[2,0], normal="Z", ind=23)
plot_vector_model(maxval,mrec_cartesian, ax=ax[2,1], normal="Z", ind=22)
plot_vector_model(maxval,mrec_cartesian, ax=ax[2,2], normal="Z", ind=21)

ax[0,0].set_title(f"y={mesh.vectorCCy[30]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[28]}")
ax[0,2].set_title(f"z={mesh.vectorCCz[27]}")
ax[1,0].set_title(f"z={mesh.vectorCCz[26]}")
ax[1,1].set_title(f"z={mesh.vectorCCz[25]}")
ax[1,2].set_title(f"z={mesh.vectorCCz[24]}")
ax[2,0].set_title(f"z={mesh.vectorCCz[23]}")
ax[2,1].set_title(f"z={mesh.vectorCCz[22]}")
ax[2,2].set_title(f"z={mesh.vectorCCz[21]}")

# %%

fig,ax = plt.subplots(3,4,figsize=(15, 10))

quiver_opts='None'
plot_amplitude(maxval,mrec_cartesian, ax=ax[0,0], ind=31)
plot_amplitude(maxval,model, ax=ax[0,1], ind=31)
plot_amplitude(maxval,mrec_cartesian, ax=ax[0,2], normal="Z", ind=27)
plot_amplitude(maxval,model, ax=ax[0,3], normal="Z", ind=27) 
plot_amplitude(maxval,mrec_cartesian, ax=ax[1,0], normal="Z", ind=25)
plot_amplitude(maxval,model, ax=ax[1,1], normal="Z", ind=25) 
plot_amplitude(maxval,mrec_cartesian, ax=ax[1,2], normal="Z", ind=24)
plot_amplitude(maxval,model, ax=ax[1,3], normal="Z", ind=24) 
plot_amplitude(maxval,mrec_cartesian, ax=ax[2,0], normal="Z", ind=22)
plot_amplitude(maxval,model, ax=ax[2,1], normal="Z", ind=22) 
plot_amplitude(maxval,mrec_cartesian, ax=ax[2,2], normal="Z", ind=20)
plot_amplitude(maxval,model, ax=ax[2,3], normal="Z", ind=20) 

ax[0,0].set_title(f"y={mesh.vectorCCy[31]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[31]}")
ax[0,2].set_title(f"z={mesh.vectorCCz[27]}")
ax[0,3].set_title(f"z={mesh.vectorCCz[27]}")
ax[1,0].set_title(f"z={mesh.vectorCCz[25]}")
ax[1,1].set_title(f"z={mesh.vectorCCz[25]}")
ax[1,2].set_title(f"z={mesh.vectorCCz[24]}")
ax[1,3].set_title(f"z={mesh.vectorCCz[24]}")
ax[2,0].set_title(f"z={mesh.vectorCCz[22]}")
ax[2,1].set_title(f"z={mesh.vectorCCz[22]}")
ax[2,2].set_title(f"z={mesh.vectorCCz[20]}")
ax[2,3].set_title(f"z={mesh.vectorCCz[20]}")


# In[60]:


## inversion in spherical coordinates
spherical_map = maps.SphericalSystem(nP=nC*3)
wires = maps.Wires(("amplitude", nC), ("theta", nC), ("phi", nC))

# create the regularization
reg_amplitude = regularization.Sparse(mesh, indActive=actv, mapping=wires.amplitude)#, alpha_s=1e-6)
reg_theta = regularization.Sparse(mesh, indActive=actv, mapping=wires.theta)#, alpha_s=1e-6)
reg_phi = regularization.Sparse(mesh, indActive=actv, mapping=wires.phi)#, alpha_s=1e-6)

norms = [[1, 0, 0, 0]] 
reg_amplitude.norms = norms
reg_theta.norms = norms
reg_phi.norms = norms

# set reference model to zero
reg_amplitude.mref = np.zeros(nC*3)
reg_theta.mref = np.zeros(nC*3)
reg_phi.mref = np.zeros(nC*3)

reg_amplitude.objfcts = reg_x.objfcts[:-1]
reg_theta.objfcts = reg_y.objfcts[:-1]
reg_phi.objfcts = reg_z.objfcts[:-1]

# don't impose reference angles
reg_theta.alpha_s = 0. 
reg_phi.alpha_s = 0.

reg_spherical = reg_amplitude + reg_theta + reg_phi


# In[62]:


simulation_spherical = mag.simulation.Simulation3DIntegral(
    mesh=mesh, survey=survey, chiMap=spherical_map, 
    actInd=actv, model_type="vector"
)

dmis_spherical = data_misfit.L2DataMisfit(simulation=simulation_spherical, data=synthetic_data)


# In[63]:

opt_spherical = optimization.InexactGaussNewton(
    maxIter=20, maxIterCG=20, tolCG=1e-4
)

# In[64]:

inv_prob_spherical = inverse_problem.BaseInvProblem(
    dmis_spherical, reg_spherical, opt_spherical, beta=inv_prob.beta
)

# In[65]:

spherical_projection = directives.ProjectSphericalBounds()  
sensitivity_weights = directives.UpdateSensitivityWeights()
IRLS = directives.Update_IRLS(
    sphericalDomain=True, beta_tol=0.1
)
update_Jacobi = directives.UpdatePreconditioner()

# In[66]:


inv_spherical = inversion.BaseInversion(
    inv_prob_spherical, directiveList=[
        spherical_projection, sensitivity_weights, IRLS, update_Jacobi
    ]
)

mstart = utils.cartesian2spherical(mrec_cartesian.reshape((nC, 3), order="F"))
mrec_spherical = inv_spherical.run(mstart)


# In[68]:

zind = 25
fig, ax = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [2, 1]})

quiver_opts = {
    "scale":np.max(np.abs(mrec_spherical))/20,
}

m = spherical_map * mrec_spherical
plot_vector_model(maxval,m, ax=ax[0,0])
plot_vector_model(maxval,m, ax=ax[0,1], normal="Z", ind=zind)
ax[0,0].set_title(f"y={mesh.vectorCCy[30]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[22]}")

quiver_opts = 'None'
plot_amplitude(maxval,m, ax=ax[1,0])
plot_amplitude(maxval,m, ax=ax[1,1], normal="Z", ind=zind)
ax[1,0].set_title(f"")
ax[1,1].set_title(f"")

plt.tight_layout()
fn = 'sparse_invmodel.png'


# In[69]:
ax = plot_data_profile(synthetic_data.dobs, plot_opts={"marker":"o", "lw":0}, label=False)
ax = plot_data_profile(inv_prob_spherical.dpred, ax=ax, label=False)#"predicted")
fn = 'sparse_inv_obs_pred.png'

# %%

fig,ax = plt.subplots(3,3,figsize=(15, 10))


quiver_opts = {
    "scale":np.max(np.abs(m))/20,
}

plot_vector_model(maxval,m, ax=ax[0,0], ind=30)
plot_vector_model(maxval,m, ax=ax[0,1], normal="Z", ind=28)
plot_vector_model(maxval,m, ax=ax[0,2], normal="Z", ind=27)
plot_vector_model(maxval,m, ax=ax[1,0], normal="Z", ind=26)
plot_vector_model(maxval,m, ax=ax[1,1], normal="Z", ind=25)
plot_vector_model(maxval,m, ax=ax[1,2], normal="Z", ind=24)
plot_vector_model(maxval,m, ax=ax[2,0], normal="Z", ind=23)
plot_vector_model(maxval,m, ax=ax[2,1], normal="Z", ind=22)
plot_vector_model(maxval,m, ax=ax[2,2], normal="Z", ind=21)

ax[0,0].set_title(f"y={mesh.vectorCCy[30]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[28]}")
ax[0,2].set_title(f"z={mesh.vectorCCz[27]}")
ax[1,0].set_title(f"z={mesh.vectorCCz[26]}")
ax[1,1].set_title(f"z={mesh.vectorCCz[25]}")
ax[1,2].set_title(f"z={mesh.vectorCCz[24]}")
ax[2,0].set_title(f"z={mesh.vectorCCz[23]}")
ax[2,1].set_title(f"z={mesh.vectorCCz[22]}")
ax[2,2].set_title(f"z={mesh.vectorCCz[21]}")


# %%
fig,ax = plt.subplots(3,4,figsize=(15, 10))

quiver_opts='None'
plot_amplitude(maxval,m, ax=ax[0,0], ind=31)
plot_amplitude(maxval,model, ax=ax[0,1], ind=31)
plot_amplitude(maxval,m, ax=ax[0,2], normal="Z", ind=27)
plot_amplitude(maxval,model, ax=ax[0,3], normal="Z", ind=27) 
plot_amplitude(maxval,m, ax=ax[1,0], normal="Z", ind=25)
plot_amplitude(maxval,model, ax=ax[1,1], normal="Z", ind=25) 
plot_amplitude(maxval,m, ax=ax[1,2], normal="Z", ind=24)
plot_amplitude(maxval,model, ax=ax[1,3], normal="Z", ind=24) 
plot_amplitude(maxval,m, ax=ax[2,0], normal="Z", ind=22)
plot_amplitude(maxval,model, ax=ax[2,1], normal="Z", ind=22) 
plot_amplitude(maxval,m, ax=ax[2,2], normal="Z", ind=20)
plot_amplitude(maxval,model, ax=ax[2,3], normal="Z", ind=20) 

ax[0,0].set_title(f"y={mesh.vectorCCy[31]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[31]}")
ax[0,2].set_title(f"z={mesh.vectorCCz[27]}")
ax[0,3].set_title(f"z={mesh.vectorCCz[27]}")
ax[1,0].set_title(f"z={mesh.vectorCCz[25]}")
ax[1,1].set_title(f"z={mesh.vectorCCz[25]}")
ax[1,2].set_title(f"z={mesh.vectorCCz[24]}")
ax[1,3].set_title(f"z={mesh.vectorCCz[24]}")
ax[2,0].set_title(f"z={mesh.vectorCCz[22]}")
ax[2,1].set_title(f"z={mesh.vectorCCz[22]}")
ax[2,2].set_title(f"z={mesh.vectorCCz[20]}")
ax[2,3].set_title(f"z={mesh.vectorCCz[20]}")






