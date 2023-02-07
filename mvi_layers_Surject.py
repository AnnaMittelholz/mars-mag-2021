#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sps
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import ipywidgets
import random
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
from pymatsolver import Pardiso as Solver


# %% Choices for different sampling scenarios 
inflight=2  # inflight sampling is 1; landed sampling is 0; 2 is a grid
use_topo=False     # topography or not? 
gauss_noise=0   # large wavelength noise on is 1.

# %% Set target Magnetization 

target_magnetization_inclination = 45
target_magnetization_declination = 90 

target_magnetization_direction = utils.mat_utils.dip_azimuth2cartesian(
    target_magnetization_inclination, target_magnetization_declination
)

target_magnetization_amplitude = 10 # magnetization in A/m
target_magnetization = target_magnetization_amplitude * target_magnetization_direction


# survey set up 
[xx, yy] = np.meshgrid(np.linspace(-600, 600, 50), np.linspace(-600, 600, 50))
b = 100
A = 50

if use_topo is True:
    zz = -A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
else: 
    zz = np.zeros_like(xx)
    
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# %% Set survey design based on scenario

if inflight == 1:  #inflight measuremenets 
    line_length = 500
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
    line_length = 500
    n_data_along_line = 15
    survey_x = np.linspace(-line_length/2, line_length/2, n_data_along_line)
    survey_y = survey_x
    survey_z = np.r_[10]
    survey_xyz = discretize.utils.ndgrid([survey_x, survey_y, survey_z])

else:
    line_length = 500
    n_data_along_line = 6
    survey_x = np.linspace(-line_length/2, line_length/2, n_data_along_line)
    survey_y = np.r_[0] 
    survey_z = np.r_[1]
    survey_xyz = discretize.utils.ndgrid([survey_x, survey_y, survey_z])


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.scatter3D(survey_xyz[:, 0], survey_xyz[:, 1], survey_xyz[:, 2], c="k", alpha=1)
plt.show()

# %% set up mesh 
from discretize import TensorMesh

nc_z = 10  # number of core mesh cells in x, y and z
dz = 3   # base cell width in x, y and z
npad_z = 10  # number of padding cells

nc = 30  # number of core mesh cells in x, y and z
dh = 15   # base cell width in x, y and z
npad = 10  # number of padding cells
exp = 1 # expansion rate of padding cells
exp_z = 1.4 # expansion rate of padding cells in z

h = [(dh, npad, -exp), (dh, nc), (dh, npad, exp)]
hz = [(dz, npad_z, -exp_z), (dz, nc_z), (dz, npad_z, exp_z)]
mesh = TensorMesh([h, h, hz], x0="CCC")

# Define an active cells from topo
actv = utils.surface2ind_topo(mesh, topo)
nC = int(actv.sum())
model_map = maps.IdentityMap(nP=nC)  # model is a vlue for each active cell

mesh
# %% Include structure in model -> here: layers

# Define the model on subsurface cells
xp = np.kron(np.ones((2)), [-100.0, 100.0, 235.0, 35.0])
yp = np.kron([-350.0, 350.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-600.0, -600.0, 45.0, 45.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind1 = utils.model_builder.PolygonInd(mesh, xyz_pts)

xp = np.kron(np.ones((2)), [-400.0, -250.0, -300.0, -150.0])
yp = np.kron([-350.0, 350.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-600.0, -600.0, 45.0, 45.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind3 = utils.model_builder.PolygonInd(mesh, xyz_pts)

magnetization = np.zeros((mesh.nC, 3))

if gauss_noise > 0:  #inflight measuremenets 

    mu, sigma = 2, 2 # mean and standard deviation
    nums = [] 

    for i in range(3*mesh.nC): 
        magnetization = random.gauss(mu, sigma)
        nums.append(magnetization) 

    magnetization = np.reshape(nums, (mesh.nC,3))

    mag_f = sp.ndimage.gaussian_filter(magnetization,2,order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    mag_f[ind1, :] = target_magnetization
    mag_f[ind3, :] = target_magnetization+4
    model_f = mag_f[actv, :]

magnetization[:, :] = -0.8*(target_magnetization)+1
magnetization[ind1, :] = target_magnetization
magnetization[ind3, :] = target_magnetization+4
model = magnetization[actv, :]

active_cell_map = maps.InjectActiveCells(mesh=mesh, indActive=actv, valInactive=np.nan)
idenMap = maps.IdentityMap(nP=nC)

# %% define plotting functions

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
            ax.plot(survey_xyz[:, 1], survey_xyz[:, 2], "C1o", ms=2,label=None)
        ax.set_xlim([survey_x.min()*1.5, survey_x.max()*1.5] if xlim is None else xlim)
      #  ax.set_ylim([target_geometry[:,2].min()*3, survey_z.max()*4] if ylim is None else ylim)
    elif normal.upper() == "Y": 
        if plot_data is True: 
            ax.plot(survey_xyz[:, 0], survey_xyz[:, 2], "C1o", ms=2,label=None)
        ax.set_xlim([survey_x.min()*1.5, survey_x.max()*1.5] if xlim is None else xlim)
      #  ax.set_ylim([target_geometry[:,2].min()*3, survey_z.max()*4] if ylim is None else ylim)
    elif normal.upper() == "Z": 
        if plot_data is True: 
            ax.plot(survey_xyz[:, 0], survey_xyz[:, 1], "C1o", ms=2,label=None)
        ax.set_xlim([survey_x.min()*1.25, survey_x.max()*1.25] if xlim is None else xlim)
        ax.set_ylim([survey_x.min()*1.25, survey_x.max()*1.25] if ylim is None else ylim)
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
        ax.set_aspect(1)
        
fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})
zind = 14
maxval=np.max(magnetization)
plot_vector_model(maxval,model, ax=ax[0],plot_data=False,plot_grid=True )
plot_vector_model(maxval,model, ax=ax[1], normal="Z", ind=zind,plot_data=False,plot_grid=True)  # APPEND WITHOUT GRID,plot_grid=False

plt.tight_layout()

# %% plot the model 

fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})
zind = 5
maxval = np.max(model)

if gauss_noise > 0: 
    plot_vector_model(maxval,model_f, ax=ax[0], normal="Y")
    plot_vector_model(maxval,model_f, ax=ax[1], normal="Z", ind=zind)  # APPEND WITHOUT GRID,plot_grid=False
    plt.tight_layout()
    fn = 'Model_Gauss.png'
    model = model_f # If Gauss nois is on overwrite model
else:   
    plot_vector_model(maxval,model, ax=ax[0])
    plot_vector_model(maxval,model, ax=ax[1], normal="Z", ind=zind)  # APPEND WITHOUT GRID,plot_grid=False
    plt.tight_layout()

fn = 'Model.png'

# %% Survey
components = ["x", "y", "z"]
rx = mag.receivers.Point(locations=survey_xyz, components=[f"b{comp}" for comp in components])

source_field = mag.sources.SourceField(
    receiver_list=[rx], parameters=np.r_[1., 0, 0]
)
survey = mag.survey.Survey(source_field)

simulation = mag.simulation.Simulation3DIntegral(
    mesh=mesh, survey=survey, chiMap=maps.IdentityMap(nP=np.prod(model.shape)), 
    actInd=actv, model_type="vector", solver=Solver
)

## % Create Synthetic Data
synthetic_data = simulation.make_synthetic_data(
    utils.mkvc(model), 
    noise_floor=0.1, # standard deviation of the noise in nT 
    relative_error=0,  # percent noise 
    add_noise=False  # do we add noise to the data we will use in the inversion?
) 
# %% Define plotting function

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

# %% Plot synthetic data 
ax = plot_data_profile(synthetic_data.dclean, label=False)
ax = plot_data_profile(synthetic_data.dobs, ax=ax, plot_opts={"marker":"o", "alpha":0.5}, label=False)
fn = 'Data_Profiles.png'

# %% Create the different domains
ind0= (~ind1) & (~ind3)
d1 = ind0[actv]
d2 = ind1[actv]
d3 = ind3[actv]

domains = [d1, d2, d3]
mapping = maps.SurjectUnits(domains, nP=3) 

#define magnetization for 3 domains corresponding to above
m1 = -0.8*(target_magnetization)+1
m2 = target_magnetization
m3 = target_magnetization+4
mags = np.array([m1,m2,m3])
mags = mags.reshape(3,3)

# provide active cell map * surjunits
mapped = mapping.P * mags
model_map = maps.IdentityMap(nP=mapping.nP)

# %% plot mapped to check it looks right and yippe it does :)     
fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})
zind = 14
maxval=np.max(magnetization)
plot_vector_model(maxval,mapped, ax=ax[0],plot_data=False,plot_grid=True )
plot_vector_model(maxval,mapped, ax=ax[1], normal="Z", ind=zind,plot_data=False,plot_grid=True)  # APPEND WITHOUT GRID,plot_grid=False

plt.tight_layout()

# %% # Create the forward model operator
prob = mag.simulation.Simulation3DIntegral(
    mesh=mesh, survey=survey, chiMap=model_map, actInd=actv,  model_type="vector", solver=Solver
)

# %% Regularization surject
regMesh = discretize.TensorMesh([len(domains)])
wires = maps.Wires(("x", nC), ("y", nC), ("z", nC))

reg_m1_x = regularization.Sparse(regMesh, mapping=wires.x)
reg_m1_y = regularization.Sparse(regMesh, mapping=wires.y)
reg_m1_z = regularization.Sparse(regMesh, mapping=wires.z)
norms = [[2, 2, 2, 2]]
reg_m1_x.norms = norms
reg_m1_y.norms = norms
reg_m1_z.norms = norms

reg = reg_m1_x + reg_m1_y + reg_m1_z

# %% 
dmis = data_misfit.L2DataMisfit(data=synthetic_data, simulation=prob)

# optimization
opt = optimization.InexactGaussNewton(
    maxIter=20, maxIterLS=20, maxIterCG=20, tolCG=1e-4
)

# inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=0)

# directives 
sensitivity_weights = directives.UpdateSensitivityWeights()  # Add sensitivity weights
IRLS = directives.Update_IRLS()  # IRLS
update_Jacobi = directives.UpdatePreconditioner()  # Pre-conditioner
target = directives.TargetMisfit(chifact=1)  # target misfit 

inv = inversion.BaseInversion(
    inv_prob, directiveList=[sensitivity_weights, IRLS, update_Jacobi, target]
)

# %% Perform inversion 
m0 = np.zeros(mapped.shape[0]*3)
m_surject = inv.run(m0)

# %% and plot L2 inversion 
zind = 5   

fig, ax = plt.subplots(2, 2, figsize=(17, 10), gridspec_kw={'width_ratios': [2, 1]})

quiver_opts = {
    "scale":np.max(np.abs(m_surject))/20,
}
maxval = np.max(np.abs(m_surject))
plot_vector_model(maxval,m_surject, ax=ax[0,0])
plot_vector_model(maxval,m_surject, ax=ax[0,1], normal="Z", ind=zind)
ax[0,0].set_title(f"y={mesh.vectorCCy[25]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[5]}")
quiver_opts='None'
plot_amplitude(maxval,m_surject, ax=ax[1,0])
plot_amplitude(maxval,m_surject, ax=ax[1,1], normal="Z", ind=zind)
ax[1,0].set_title(f"")
ax[1,1].set_title(f"")
plt.tight_layout()
fn = 'L2_invmodel.png'

# %%More depth profiles
fig,ax = plt.subplots(3,3,figsize=(15, 10))


quiver_opts = {
    "scale":np.max(np.abs(m_surject))/20,
}

plot_vector_model(maxval,m_surject, ax=ax[0,0], ind=30)
plot_vector_model(maxval,m_surject, ax=ax[0,1], normal="Z", ind=14)
plot_vector_model(maxval,m_surject, ax=ax[0,2], normal="Z", ind=12)
plot_vector_model(maxval,m_surject, ax=ax[1,0], normal="Z", ind=10)
plot_vector_model(maxval,m_surject, ax=ax[1,1], normal="Z", ind=8)
plot_vector_model(maxval,m_surject, ax=ax[1,2], normal="Z", ind=6)
plot_vector_model(maxval,m_surject, ax=ax[2,0], normal="Z", ind=4)
plot_vector_model(maxval,m_surject, ax=ax[2,1], normal="Z", ind=2)
plot_vector_model(maxval,m_surject, ax=ax[2,2], normal="Z", ind=1)

ax[0,0].set_title(f"Fwd Model at y={mesh.vectorCCz[10]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[14]}")
ax[0,2].set_title(f"z={mesh.vectorCCz[12]}")
ax[1,0].set_title(f"z={mesh.vectorCCz[10]}")
ax[1,1].set_title(f"z={mesh.vectorCCz[8]}")
ax[1,2].set_title(f"z={mesh.vectorCCz[6]}")
ax[2,0].set_title(f"z={mesh.vectorCCz[4]}")
ax[2,1].set_title(f"z={mesh.vectorCCz[2]}")
ax[2,2].set_title(f"z={mesh.vectorCCz[1]}")

# %%

fig,ax = plt.subplots(3,4,figsize=(15, 10))

quiver_opts='None'
plot_amplitude(maxval,m_surject, ax=ax[0,0], ind=14)
plot_amplitude(maxval,model, ax=ax[0,1], ind=14)
plot_amplitude(maxval,m_surject, ax=ax[0,2], normal="Z", ind=13)
plot_amplitude(maxval,model, ax=ax[0,3], normal="Z", ind=13) 
plot_amplitude(maxval,m_surject, ax=ax[1,0], normal="Z", ind=12)
plot_amplitude(maxval,model, ax=ax[1,1], normal="Z", ind=12) 
plot_amplitude(maxval,m_surject, ax=ax[1,2], normal="Z", ind=10)
plot_amplitude(maxval,model, ax=ax[1,3], normal="Z", ind=10) 
plot_amplitude(maxval,m_surject, ax=ax[2,0], normal="Z", ind=8)
plot_amplitude(maxval,model, ax=ax[2,1], normal="Z", ind=8) 
plot_amplitude(maxval,m_surject, ax=ax[2,2], normal="Z", ind=6)
plot_amplitude(maxval,model, ax=ax[2,3], normal="Z", ind=6) 

ax[0,0].set_title(f"y={mesh.vectorCCy[14]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[14]}")
ax[0,2].set_title(f"z={mesh.vectorCCz[13]}")
ax[0,3].set_title(f"z={mesh.vectorCCz[13]}")
ax[1,0].set_title(f"z={mesh.vectorCCz[12]}")
ax[1,1].set_title(f"z={mesh.vectorCCz[12]}")
ax[1,2].set_title(f"z={mesh.vectorCCz[10]}")
ax[1,3].set_title(f"z={mesh.vectorCCz[10]}")
ax[2,0].set_title(f"z={mesh.vectorCCz[8]}")
ax[2,1].set_title(f"z={mesh.vectorCCz[8]}")
ax[2,2].set_title(f"z={mesh.vectorCCz[6]}")
ax[2,3].set_title(f"z={mesh.vectorCCz[6]}")


# In[22]:

ax = plot_data_profile(synthetic_data.dobs, plot_opts={"marker":"o", "lw":0}, label="observed")
ax = plot_data_profile(inv_prob.dpred, ax=ax, label=False)
 
fn = 'L2_inv_obs_pred.png'
# %% Some DATA plots

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
