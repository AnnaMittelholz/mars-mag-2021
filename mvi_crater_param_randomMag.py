#!/usr/bin/env python
# coding: utf-8

# %%Load packages
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

inflight=0  # inflight sampling is 1; landed sampling is 0; 2 is a grid
use_topo=True     # topography or not? 
gauss_noise=0   # large wavelength noise on is 1.


# %% Magnetization 
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
# ax.plot_trisurf(
#     topo[:, 0], topo[:, 1], topo[:, 2], triangles=tri.simplices, cmap=plt.cm.Spectral
# )
ax.scatter3D(survey_xyz[:, 0], survey_xyz[:, 1], survey_xyz[:, 2], c="k", alpha=1)
plt.show()


# %% set up mesh 


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


# %% Magnetization of target 


ind = utils.model_builder.getIndicesSphere([0,0,-40], 120, mesh.gridCC)

magnetization = np.zeros((mesh.nC, 3))

if gauss_noise > 0:  #inflight measuremenets 

    mu, sigma = 2, 2 # mean and standard deviation
    nums = [] 

    for i in range(3*mesh.nC): 
        magnetization = random.gauss(mu, sigma)
        nums.append(magnetization) 

    magnetization = np.reshape(nums, (mesh.nC,3))

    mag_f = sp.ndimage.gaussian_filter(magnetization,2,order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    mag_f[ind, :] = target_magnetization
    model_f = mag_f[actv, :]

magnetization[ind, :] = target_magnetization
model = magnetization[actv, :]

active_cell_map = maps.InjectActiveCells(mesh=mesh, indActive=actv, valInactive=np.nan)


# In[10]:


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
        

# %% plot the model 
fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})
zind = 27
maxval=target_magnetization_amplitude
plot_vector_model(maxval,model, ax=ax[0],plot_data=False )
plot_vector_model(maxval,model, ax=ax[1], normal="Z", ind=zind,plot_data=False)  # APPEND WITHOUT GRID,plot_grid=False

plt.tight_layout()


# %% plot the model 
maxval = target_magnetization_amplitude
fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})
zind = 25

if gauss_noise > 0: 
    
    maxval=target_magnetization_amplitude
    plot_vector_model(maxval,model_f, ax=ax[0], normal="Y")
    plot_vector_model(maxval,model_f, ax=ax[1], normal="Z", ind=zind)  # APPEND WITHOUT GRID,plot_grid=False

    plt.tight_layout()

    fn = 'Model_Gauss.png'
    
    
maxval=target_magnetization_amplitude
plot_vector_model(maxval,model, ax=ax[0])
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
    actInd=actv, model_type="vector", solver=Solver
)


# In[16]:


synthetic_data = simulation.make_synthetic_data(
    utils.mkvc(model), 
    noise_floor=0.1, # standard deviation of the noise in nT 
    relative_error=0,  # percent noise 
    add_noise=True  # do we add noise to the data we will use in the inversion?
) 


# In[17]:


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

ax = plot_data_profile(synthetic_data.dclean, label="clean")
ax = plot_data_profile(synthetic_data.dobs, ax=ax, plot_opts={"marker":"o", "alpha":0.5}, label=False)
 
fn = 'Data_Profiles.png'


# %% create the regularization

wires = maps.Wires(("x", nC), ("y", nC), ("z", nC))

reg_x = regularization.Sparse(mesh, indActive=actv, mapping=wires.x, alpha_s=1e-4, alpha_z=1e-8)
reg_y = regularization.Sparse(mesh, indActive=actv, mapping=wires.y, alpha_s=1e-4, alpha_z=1e-8)
reg_z = regularization.Sparse(mesh, indActive=actv, mapping=wires.z, alpha_s=1e-4, alpha_z=1e-8)

norms = [[2, 2, 2, 2]]
reg_x.norms = norms
reg_y.norms = norms
reg_z.norms = norms

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
betaest = directives.BetaEstimate_ByEig(beta0_ratio=4)  # estimate initial trade-off parameter
sensitivity_weights = directives.UpdateSensitivityWeights()  # Add sensitivity weights
IRLS = directives.Update_IRLS()  # IRLS
update_Jacobi = directives.UpdatePreconditioner()  # Pre-conditioner
target = directives.TargetMisfit(chifact=1)  # target misfit 


inv = inversion.BaseInversion(
    inv_prob, directiveList=[sensitivity_weights, IRLS, update_Jacobi, betaest, target]
)


# %% Perform inversion 

m0 = np.zeros(nC * 3)
mrec_cartesian = inv.run(m0)

# %% and plot L2 inversion 

maxval =target_magnetization_amplitude /2
fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})

quiver_opts = {
    "scale":np.max(np.abs(mrec_cartesian))/20,
}

plot_vector_model(maxval,mrec_cartesian, ax=ax[0], ind=30)
plot_vector_model(maxval,mrec_cartesian, ax=ax[1], normal="Z", ind=27)
ax[0].set_title(f"y={mesh.vectorCCy[30]}")
ax[1].set_title(f"z={mesh.vectorCCz[27]}")

#plot_target_outline(ax[0], normal="Y")
#plot_target_outline(ax[1], normal="Z")

fn = 'L2_invmodel.png'
#plt.savefig(pn+fn)


# In[22]:


ax = plot_data_profile(synthetic_data.dobs, plot_opts={"marker":"o", "lw":0}, label="observed")
ax = plot_data_profile(inv_prob.dpred, ax=ax, label=False)
 
fn = 'L2_inv_obs_pred.png'

# %% 
# NOW THIS IS THE NEW PART -> TRY PARAMETERIZED INVERSION 
# # hemispherical mapping
# #


class MagnetizedSphere(maps.BaseParametric):
    """
    Mapping for a sphere of radius R in a wholespace 
    
    m = [x, y, z, r, mx, my, mz]
    
    is the position, radius and magnetization in each direction 
    """
        
    @property
    def nP(self):
        return 7

    @property
    def shape(self):
        if self.indActive is not None: 
            return (self.indActive.sum()*3, self.nP)
        return (self.mesh.nC * 3, self.nP)
    
    def _root_distance(self, m):
        x, y, z = m[0], m[1], m[2]
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2 + (self.z - z)**2)
    
    def _sphere(self, m):
        r = m[3]
        sphere = r - self._root_distance(m)
        return sphere

    def _sphere_deriv_x(self, m): 
        x, y, z = m[0], m[1], m[2]
        return -(self.x - x) / self._root_distance(m)
    
    def _sphere_deriv_y(self, m): 
        x, y, z = m[0], m[1], m[2]
        return -(self.y - y) / self._root_distance(m)
    
    def _sphere_deriv_z(self, m): 
        x, y, z = m[0], m[1], m[2]
        return -(self.z - z) / self._root_distance(m)
    
    def _sphere_deriv_r(self, m): 
        return -np.ones(self.indActive.sum())

    def _transform(self, m):
        mx, my, mz = m[4], m[5], m[6]
        sphere = self._atanfct(self._sphere(m), self.slope)
        return np.hstack([mx * sphere, my * sphere, mz * sphere])

    def deriv(self, m, v=None):
        mx, my, mz = m[4], m[5], m[6]
        zeros = np.zeros(self.indActive.sum())
        
        atanfctderiv = self._atanfctDeriv(self._sphere(m), self.slope)
        
        sphere_deriv_x = atanfctderiv * self._sphere_deriv_x(m)
        deriv_x = np.hstack([mx * sphere_deriv_x, my * sphere_deriv_x, mz * sphere_deriv_x]) 
        
        sphere_deriv_y = atanfctderiv * self._sphere_deriv_y(m)
        deriv_y = np.hstack([mx * sphere_deriv_y, my * sphere_deriv_y, mz * sphere_deriv_y])
        
        sphere_deriv_z = atanfctderiv * self._sphere_deriv_z(m)
        deriv_z = np.hstack([mx * sphere_deriv_z, my * sphere_deriv_z, mz * sphere_deriv_z])
        
        dr = self._atanfctDeriv(self._sphere(m), self.slope) * self._sphere_deriv_r(m)
        deriv_r = np.hstack([mx*dr, my*dr, mz*dr])
        
        sphere = self._atanfct(self._sphere(m), self.slope)
        deriv_mx = np.hstack([sphere, zeros, zeros])
        deriv_my = np.hstack([zeros, sphere, zeros])
        deriv_mz = np.hstack([zeros, zeros, sphere])
        
        deriv = sps.csr_matrix(np.c_[
            deriv_x, deriv_y, deriv_z, deriv_r, deriv_mx, deriv_my, deriv_mz
        ])
        
        if v is not None:
            return deriv * v
        return deriv


# %% sphere 
sphere_map = MagnetizedSphere(mesh, indActive=active_cell_map.indActive, slope=5e-1)

mtest = np.r_[0, 0, 0, 120, 7, 0, -7]  #m = [x, y, z, r, mx, my, mz]

np.abs(sphere_map * mtest)[:actv.sum()].min()


# PLOT
maxval =5
fig, ax = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [2, 1]})

maxval=target_magnetization_amplitude
plot_vector_model(maxval, sphere_map * mtest, ax=ax[0,0])
plot_vector_model(maxval, sphere_map * mtest, ax=ax[0,1], normal="Z", ind=zind)  # APPEND WITHOUT GRID,plot_grid=False

ax[0,0].set_title(f"y={mesh.vectorCCy[30]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[zind]}")

quiver_opts = 'None'
plot_amplitude(maxval,sphere_map * mtest, ax=ax[1,0])
plot_amplitude(maxval,sphere_map * mtest, ax=ax[1,1], normal="Z", ind=zind)
ax[1,0].set_title(f"")
ax[1,1].set_title(f"")

plt.tight_layout()


# %% What does test do?

print(sphere_map.test(mtest))


# %% set up again


simulation_parametric = mag.simulation.Simulation3DIntegral(
    mesh=mesh, survey=survey, chiMap=sphere_map, 
    actInd=actv, model_type="vector", solver=Solver
)


# %% create the regularization
reg = regularization.SimpleSmall(mesh=discretize.TensorMesh([sphere_map.nP]))

# simulation2.G dimensions mismatch 
dmis = data_misfit.L2DataMisfit(data=synthetic_data, simulation=simulation_parametric)

# optimization
opt = optimization.ProjectedGNCG(
    maxIter=20, maxIterLS=20, maxIterCG=20, tolCG=1e-4,
    lower=np.r_[-np.inf, -np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf]
)

# inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=0)  

# directives 
target = directives.TargetMisfit(chifact=1)  # target misfit LH 0

inv = inversion.BaseInversion(
    inv_prob, directiveList=[target]
)


# %% run iversion


m0_parametric = np.r_[0, 0, 0, 100, 1, 1, 1]  # note: can't start at all zeros, it will fail to find a search direction
mrec_parametric = inv.run(m0_parametric)


print(mrec_parametric)


# In[31]:


fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={'width_ratios': [2, 1]})

maxval=target_magnetization_amplitude
plot_vector_model(maxval, sphere_map * mrec_parametric, ax=ax[0])
plot_vector_model(maxval, sphere_map * mrec_parametric, ax=ax[1], normal="Z", ind=zind)  # APPEND WITHOUT GRID,plot_grid=False

ax[0,0].set_title(f"y={mesh.vectorCCy[30]}")
ax[0,1].set_title(f"z={mesh.vectorCCz[22]}")

quiver_opts = 'None'
plot_amplitude(maxval,sphere_map * mrec_parametric, ax=ax[1,0])
plot_amplitude(maxval,sphere_map * mrec_parametric, ax=ax[1,1], normal="Z", ind=zind)
ax[1,0].set_title(f"")
ax[1,1].set_title(f"")

plt.tight_layout()





