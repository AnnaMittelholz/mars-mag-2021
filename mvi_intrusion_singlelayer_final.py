#!/usr/bin/env python
# coding: utf-8
# general python ecosystem
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import ipywidgets
import random

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

from pymatsolver import Pardiso as Solver


# %% Choices for different sampling scenarios 

inflight=2  # inflight sampling is 1; landed sampling is 0; 2 is a grid
use_topo=False     # topography or not? 
gauss_noise=0   # large wavelength noise on is 1.


# %% Magnetization 


target_magnetization_inclination = 45
target_magnetization_declination = 90 

target_magnetization_direction = utils.mat_utils.dip_azimuth2cartesian(
    target_magnetization_inclination, target_magnetization_declination
)

target_magnetization_amplitude = 10 # magnetization in A/m
target_magnetization = target_magnetization_amplitude * target_magnetization_direction


# %% survey set up 

[xx, yy] = np.meshgrid(np.linspace(-600, 600, 50), np.linspace(-600, 600, 50))
b = 100
A = 50

if use_topo is True:
    zz = -A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
else: 
    zz = np.zeros_like(xx)
    
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]


# In[6]:


if inflight == 1:  #inflight measuremenets 
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



# %% make mesh

# Create a mesh
from discretize import TensorMesh


nc_z = 5  # number of core mesh cells in x, y and z
dz = 10   # base cell width in x, y and z
npad_z = 10  # number of padding cells


nc = 40  # number of core mesh cells in x, y and z
dh = 20   # base cell width in x, y and z
npad = 10  # number of padding cells
exp = 1  # expansion rate of padding cells

h = [(dh, npad, -exp), (dh, nc), (dh, npad, exp)]
hz = [(dz, npad_z, -exp), (dz, nc_z), (dz, npad_z, exp)]
mesh = TensorMesh([h, h, hz], x0="CCC")


# Define an active cells from topo
actv = utils.surface2ind_topo(mesh, topo)
nC = int(actv.sum())
model_map = maps.IdentityMap(nP=nC)  # model is a vlue for each active cell

mesh

# %%  make Geometry of magnetized body  


# Define the model on subsurface cells
# Add dyke number 1
xp = np.kron(np.ones((2)), [-100.0, 100.0, 235.0, 35.0])
yp = np.kron([-350.0, 350.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-600.0, -600.0, 45.0, 45.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind1 = utils.model_builder.PolygonInd(mesh, xyz_pts)

xp = np.kron(np.ones((2)), [100.0, 200.0, 335.0, 235.0])
yp = np.kron([-350.0, 350.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-600.0, -600.0, 45.0, 45.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind2 = utils.model_builder.PolygonInd(mesh, xyz_pts)

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
    mag_f[ind2, :] = target_magnetization+50
    mag_f[ind3, :] = target_magnetization+40
    model_f = mag_f[actv, :]

magnetization[ind1, :] = target_magnetization
magnetization[ind2, :] = target_magnetization+50
magnetization[ind3, :] = target_magnetization+40
model = magnetization[actv, :]

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
    
    
# %%  Plot Model

fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})
zind = 9
maxval=target_magnetization_amplitude+50
plot_vector_model(maxval,model, ax=ax[0], plot_data=False, plot_grid=True)
plot_vector_model(maxval,model, ax=ax[1], normal="Z", ind=zind, plot_data=False)  # APPEND WITHOUT GRID,plot_grid=False

plt.tight_layout()


# %% plot the model 

fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})
zind = 9

if gauss_noise > 0: 
    
    maxval=target_magnetization_amplitude
    plot_vector_model(maxval,model_f, ax=ax[0], normal="Y")
    plot_vector_model(maxval,model_f, ax=ax[1], normal="Z", ind=zind)  # APPEND WITHOUT GRID,plot_grid=False

    plt.tight_layout()

    fn = 'Model_Gauss.png'
    
    

plot_vector_model(maxval,model, ax=ax[0], plot_grid=True)
plot_vector_model(maxval,model, ax=ax[1], normal="Z", ind=zind)  # APPEND WITHOUT GRID,plot_grid=False

plt.tight_layout()

fn = 'Model.png'


# %% If Gauss nois is on overwrite model

if gauss_noise > 0: 
    model = model_f

# %%


def plot_target_outline(ax, normal="Y", plot_opts={"color":"C3", "lw":3}):
    if normal.upper() == "X": 
        x_target = np.hstack([target_geometry[:, 1], target_geometry[::-1, 1], target_geometry[0, 1]])
        y_target = np.hstack([target_geometry[0, 2], target_geometry[:, 2], target_geometry[::-1, 2]])
    elif normal.upper() == "Y": 
        x_target = np.hstack([target_geometry[:, 0], target_geometry[::-1, 0], target_geometry[0, 0]])
        y_target = np.hstack([target_geometry[0, 2], target_geometry[:, 2], target_geometry[::-1, 2]])
    elif normal.upper() == "Z":
        x_target = np.hstack([target_geometry[:, 0], target_geometry[::-1, 0], target_geometry[0, 0]])
        y_target = np.hstack([target_geometry[0, 1], target_geometry[:, 1], target_geometry[::-1, 1]])
    
    ax.plot(x_target, y_target, **plot_opts)

# %% Survey

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

ax = plot_data_profile(synthetic_data.dclean, label=False)
ax = plot_data_profile(synthetic_data.dobs, ax=ax, plot_opts={"marker":"o", "alpha":0.5},label=False)
 
fn = 'Data_Profiles.png'


# %% NOW SINGLE LAYER ACTION STARTS
#  make new mesh for the single layer

ncz = 2  # number of core mesh cells in z
dz = 100  # base cell width z
hz = dz * np.ones(ncz)


nc = 40  # number of core mesh cells in x, y and z
dh = 20   # base cell width in x, y and z
npad = 10  # number of padding cells
exp = 1  # expansion rate of padding cells

h = [(dh, npad, -exp), (dh, nc), (dh, npad, exp)]
mesh = TensorMesh([h, h, hz], x0="CCC")

# Define an active cells from topo
actv = utils.surface2ind_topo(mesh, topo)
nC = int(actv.sum())
model_map = maps.IdentityMap(nP=nC)  # model is a vlue for each active cell

mesh
# %% The 1 layer magnetized model 
xp = np.kron(np.ones((2)), [-100.0, 100.0, 235.0, 35.0])
yp = np.kron([-350.0, 350.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-600.0, -600.0, 45.0, 45.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind1 = utils.model_builder.PolygonInd(mesh, xyz_pts)

xp = np.kron(np.ones((2)), [100.0, 200.0, 335.0, 235.0])
yp = np.kron([-350.0, 350.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-600.0, -600.0, 45.0, 45.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind2 = utils.model_builder.PolygonInd(mesh, xyz_pts)

xp = np.kron(np.ones((2)), [-400.0, -250.0, -300.0, -150.0])
yp = np.kron([-350.0, 350.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-600.0, -600.0, 45.0, 45.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind3 = utils.model_builder.PolygonInd(mesh, xyz_pts)



magnetization_1l = np.zeros((mesh.nC, 3))

if gauss_noise > 0:  #inflight measuremenets 

    mu, sigma = 2, 2 # mean and standard deviation
    nums = [] 

    for i in range(3*mesh.nC): 
        magnetization = random.gauss(mu, sigma)
        nums.append(magnetization) 

    magnetization = np.reshape(nums, (mesh.nC,3))

    mag_f = sp.ndimage.gaussian_filter(magnetization,2,order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    mag_f[ind1, :] = target_magnetization
    mag_f[ind2, :] = target_magnetization+50
    mag_f[ind3, :] = target_magnetization+40
    model_f = mag_f[actv, :]

magnetization_1l[ind1, :] = target_magnetization
magnetization_1l[ind2, :] = target_magnetization+50
magnetization_1l[ind3, :] = target_magnetization+40
model_1l = magnetization_1l[actv, :]

active_cell_map = maps.InjectActiveCells(mesh=mesh, indActive=actv, valInactive=np.nan)
# %%
fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})

zind = 0

plot_vector_model(maxval,model_1l, ax=ax[0], plot_grid=True)
plot_vector_model(maxval,model_1l, ax=ax[1], normal="Z", ind=zind)  # APPEND WITHOUT GRID,plot_grid=False
plt.tight_layout()

fn = 'Model2.png'

# %%
# create the regularization
wires = maps.Wires(("x", nC), ("y", nC), ("z", nC))

reg_x = regularization.Sparse(mesh, indActive=actv, mapping=wires.x)#, alpha_s=1e-4, alpha_z=1e-8)
reg_y = regularization.Sparse(mesh, indActive=actv, mapping=wires.y)#, alpha_s=1e-4, alpha_z=1e-8)
reg_z = regularization.Sparse(mesh, indActive=actv, mapping=wires.z)#, alpha_s=1e-4, alpha_z=1e-8)

norms = [[2, 2, 2, 2]]
reg_x.norms = norms
reg_y.norms = norms
reg_z.norms = norms

reg_x.objfcts = reg_x.objfcts[:-1]
reg_y.objfcts = reg_y.objfcts[:-1]
reg_z.objfcts = reg_z.objfcts[:-1]

reg = reg_x + reg_y + reg_z

# data misfit
simulation2 = mag.simulation.Simulation3DIntegral(
    mesh=mesh, survey=survey, chiMap=maps.IdentityMap(nP=np.prod(model_1l.shape)), #this must be wrong!
    actInd=actv, model_type="vector"
)
# simulation2.G dimensions mismatch 
dmis = data_misfit.L2DataMisfit(data=synthetic_data, simulation=simulation2)

# optimization
opt = optimization.InexactGaussNewton(
    maxIter=10, maxIterLS=10, maxIterCG=20, tolCG=1e-4
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

 # %% run the inversion

m0 = np.zeros(nC * 3)
mrec_cartesian = inv.run(m0)

# %% 


fig, ax = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={'width_ratios': [2, 1]})

quiver_opts = {
    "scale":np.max(np.abs(mrec_cartesian))/20,
}
maxval = target_magnetization_amplitude+30
plot_vector_model(maxval,mrec_cartesian, ax=ax[0])
plot_vector_model(maxval,mrec_cartesian, ax=ax[1], normal="Z",ind=0)
ax[0].set_title(f"y={mesh.vectorCCy[30]}")
ax[1].set_title(f"z={mesh.vectorCCz[0]}")

fn = 'L2_invmodel.png'
#plt.savefig(pn+fn)

# %%
# Plotting

fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})

zind = 0
ax = plot_data_profile(synthetic_data.dobs, plot_opts={"marker":"o", "lw":0},  label=False)
ax = plot_data_profile(inv_prob.dpred, ax=ax, label=False)
 
fn = 'L2_inv_obs_pred.png'
#plt.savefig(pn+fn)


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


# directives 
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

zind = 0
fig, ax = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={'width_ratios': [2, 1]})

quiver_opts = {
    "scale":np.max(np.abs(mrec_spherical))/20,
}

m = spherical_map * mrec_spherical
plot_vector_model(40,m, ax=ax[0])
plot_vector_model(40,m, ax=ax[1], normal="Z", ind=zind)

#plot_target_outline(ax[0], normal="Y")
#plot_target_outline(ax[1], normal="Z")
fn = 'sparse_invmodel.png'
#plt.savefig(pn+fn)


# In[69]:

ax = plot_data_profile(synthetic_data.dobs, plot_opts={"marker":"o", "lw":0}, label=False)
ax = plot_data_profile(inv_prob_spherical.dpred, ax=ax, label=False)

fn = 'sparse_inv_obs_pred.png'
#plt.savefig(pn+fn)


