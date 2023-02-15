#!/usr/bin/env python
# coding: utf-8
# general python ecosystem

# %%
# general python ecosystem
import numpy as np
import scipy as sp
import scipy.sparse as sps
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge, Rectangle
import ipywidgets
import random
import math
from matplotlib import patches

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
import os
# %% Load in model iterations
inflight=2    # inflight sampling is 1; landed sampling is 0; 2 is mesh
Itype = 'sparse'

if inflight == 1:
    path = './model_iterations/Layers_'+Itype+'/flight/'
elif inflight==0:
    path = './model_iterations/Layers_'+Itype+'/landed/'
elif inflight==2:
    path = './model_iterations/Layers_'+Itype+'/mesh/'


# %% Set up the code
 

use_topo=False 

target_magnetization_inclination = 45
target_magnetization_declination = 90 

target_magnetization_direction = utils.mat_utils.dip_azimuth2cartesian(
    target_magnetization_inclination, target_magnetization_declination
)

target_magnetization_amplitude = 10 # magnetization in A/m
background_magnetization = 0 # magnetization in A/m
target_magnetization = target_magnetization_amplitude * target_magnetization_direction

# define Topography if flag is set to true
[xx, yy] = np.meshgrid(np.linspace(-400, 400, 50), np.linspace(-400, 400, 50))
b = 100
A = 50

if use_topo is True:
    zz = -A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
else: 
    zz = np.zeros_like(xx)
    
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# %%
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
    # %%
from discretize import TensorMesh

nc_z = 10#5  # number of core mesh cells in x, y and z
dz = 3   # base cell width in x, y and z
npad_z = 10  # number of padding cells

nc = 30  # number of core mesh cells in x, y and z
dh = 15   # base cell width in x, y and z
npad = 10  # number of padding cells
exp = 1 # expansion rate of padding cells
exp_z = 1.4 # was 1.4

h = [(dh, npad, -exp), (dh, nc), (dh, npad, exp)]
hz = [(dz, npad_z, -exp_z), (dz, nc_z), (dz, npad_z, exp_z)]
mesh = TensorMesh([h, h, hz], x0="CCC")

# Define an active cells from topo
actv = utils.surface2ind_topo(mesh, topo)
nC = int(actv.sum())
model_map = maps.IdentityMap(nP=nC)  # model is a vlue for each active cell

mesh

# %%
xp = np.kron(np.ones((2)), [-100.0, 100.0, 235.0, 35.0])
yp = np.kron([-350.0, 350.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-600.0, -600.0, 45.0, 45.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind1 = utils.model_builder.PolygonInd(mesh, xyz_pts)
print(xyz_pts)
print(xp)
xp = np.kron(np.ones((2)), [-400.0, -250.0, -300.0, -150.0])
yp = np.kron([-350.0, 350.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-600.0, -600.0, 45.0, 45.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind3 = utils.model_builder.PolygonInd(mesh, xyz_pts)

print(ind3)

magnetization = np.zeros((mesh.nC, 3))
magnetization[:, :] = -0.8*(target_magnetization)+1
magnetization[ind1, :] = target_magnetization
magnetization[ind3, :] = target_magnetization+4
model = magnetization[actv, :]
active_cell_map = maps.InjectActiveCells(mesh=mesh, indActive=actv, valInactive=np.nan)
# %%
def full_mesh_magnetization(model,nC,active_cell_map):
    return np.vstack([active_cell_map * model.reshape(nC, 3, order="F")[:, i] for i in range(3)]).T

def plot_vector_model(
    mesh, nC, active_cell_map, maxval, model, ax=None, quiver_opts=None, normal="Y", xlim=None, ylim=None, ind=None, plot_data=True, plot_grid=False, outline=True
):
    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    magnetization = full_mesh_magnetization(model,nC,active_cell_map)
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
            grid=plot_grid, ind=ind, quiver_opts=qo,label=False
        )[0], ax=ax
    )
    
    cb.set_label("amplitude magnetization (A/m)", fontsize=14)
    rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
    
    if normal.upper() == "X": 
        if plot_data is True: 
            ax.plot(survey_xyz[:, 1], survey_xyz[:, 2], "C1o", ms=4,label=False)
        ax.set_xlim([survey_x.min()*1.5, survey_x.max()*1.1] if xlim is None else xlim)
        ax.set_title(f"x at {mesh.vectorCCy[ind]} m", fontsize=14)
        if outline is True:
                ax.plot((-345, -315), (-310, 0), linestyle="dashed",linewidth=2,color='k')
                ax.plot((-195, -165), (-310, 0), linestyle="dashed",linewidth=2,color='k')
                ax.plot((-28, 30), (-310, 0), linestyle="dashed",linewidth=2,color='k')
                ax.plot((170, 225), (-310, 0), linestyle="dashed",linewidth=2,color='k')
    elif normal.upper() == "Y": 
        if plot_data is True: 
            ax.plot(survey_xyz[:, 0], survey_xyz[:, 2], "C1o", ms=4,label=False)
        ax.set_xlim([survey_x.min()*1.5, survey_x.max()*1.5] if xlim is None else xlim)
        ax.set_title("y at 0 m", fontsize=14)
        if outline is True:
                ax.plot((-345, -315), (-310, 0), linestyle="dashed",linewidth=2,color='r')
                ax.plot((-195, -165), (-310, 0), linestyle="dashed",linewidth=2,color='r')
                ax.plot((-28, 30), (-310, 0), linestyle="dashed",linewidth=2,color='r')
                ax.plot((170, 225), (-310, 0), linestyle="dashed",linewidth=2,color='r')
    elif normal.upper() == "Z": 
        if plot_data is True: 
            ax.plot(survey_xyz[:, 0], survey_xyz[:, 1], "C1o", ms=4,label=False)
        ax.set_xlim([survey_x.min()*1.25, survey_x.max()*1.25] if xlim is None else xlim)
        ax.set_ylim([survey_x.min()*1.25, survey_x.max()*1.25] if ylim is None else ylim)
        ax.set_title(f"z at {int(mesh.vectorCCz[ind])} m", fontsize=14)
        if outline is True:
            plt.axvline(x=-164, ymin=-310, ymax= 310, linewidth=2, color='r',linestyle="dashed")
            plt.axvline(x=211, ymin=-310, ymax= 310, linewidth=2, color='r',linestyle="dashed")
            plt.axvline(x=16, ymin=-310, ymax= 310, linewidth=2, color='r',linestyle="dashed")
    ax.set_aspect(1)
    
    
def plot_amplitude(
        mesh, nC, active_cell_map, maxval, model, ax=None, quiver_opts=None, normal="Y", xlim=None, ylim=None, ind=None, plot_data=True, plot_grid=False, outline=True
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
                
        magnetization = full_mesh_magnetization(model,nC,active_cell_map)
        Mtotal = np.sqrt(magnetization[:,0]**2 + magnetization[:,1]**2 + magnetization[:,2]**2)
                
        cb = plt.colorbar(
            mesh.plot_slice(
                Mtotal, "CC", clim=[0,maxval],normal=normal, ax=ax, view="abs", 
                grid=plot_grid, ind=ind, quiver_opts=qo
                )[0], ax=ax
            )
        
        cb.set_label("amplitude magnetization [A/m]", fontsize=14)

        if normal.upper() == "X": 
            if plot_data is True: 
                ax.plot(survey_xyz[:, 1], survey_xyz[:, 2], "C1o", ms=4,label=False)
            ax.set_xlim([survey_x.min()*1.5, survey_x.max()*1.5] if xlim is None else xlim)
            ax.set_title(f"x at {mesh.vectorCCy[ind]} m", fontsize=14)
            if outline is True:
                ax.plot((-345, -305), (-310, 0), linestyle="dashed",linewidth=2,color='k')
                ax.plot((-195, -170), (-310, 0), linestyle="dashed",linewidth=2,color='k')
                ax.plot((-25, 15), (-310, 0), linestyle="dashed",linewidth=2,color='k')
        elif normal.upper() == "Y": 
            if plot_data is True: 
                ax.plot(survey_xyz[:, 0], survey_xyz[:, 2], "C1o", ms=4)
            ax.set_xlim([survey_x.min()*1.5, survey_x.max()*1.5] if xlim is None else xlim)
            ax.set_title("y at 0 m", fontsize=14)
            if outline is True:
                ax.plot((-345, -315), (-310, 0), linestyle="dashed",linewidth=2,color='k')
                ax.plot((-195, -165), (-310, 0), linestyle="dashed",linewidth=2,color='k')
                ax.plot((-28, 30), (-310, 0), linestyle="dashed",linewidth=2,color='k')
                ax.plot((170, 225), (-310, 0), linestyle="dashed",linewidth=2,color='k')
        elif normal.upper() == "Z": 
            if plot_data is True: 
                ax.plot(survey_xyz[:, 0], survey_xyz[:, 1], "C1o", ms=4)
            ax.set_xlim([survey_x.min()*1.25, survey_x.max()*1.25] if xlim is None else xlim)
            ax.set_ylim([survey_x.min()*1.25, survey_x.max()*1.25] if ylim is None else ylim)
            ax.set_title(f"z at {int(mesh.vectorCCz[ind])} m", fontsize=14)
            if outline is True:
                ax.axvline(x=-164, ymin=-310, ymax= 310, linewidth=2, color='k',linestyle="dashed")
                ax.axvline(x=211, ymin=-310, ymax= 310, linewidth=2, color='k',linestyle="dashed")
                ax.axvline(x=16, ymin=-310, ymax= 310, linewidth=2, color='k',linestyle="dashed")

        ax.set_aspect(1)


# %%

#for last 5 files in path read in data and plot
zind = 5
spherical_map = maps.SphericalSystem(nP=nC*3)

for file in os.listdir(path):
    model = np.load(path+file,allow_pickle=True)

    fig, ax = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [2, 1]})

    quiver_opts = {
        "scale":np.max(np.abs(model))/20,
    }
    if Itype =='sparse':
        m = spherical_map * model
    else: 
        m = model

    maxval = 11#target_magnetization_amplitude

    plot_vector_model(mesh, nC, active_cell_map,maxval,m, ax=ax[0,0])
    plot_vector_model(mesh, nC, active_cell_map,maxval,m, ax=ax[0,1], normal="Z", ind=zind)
    plt.tight_layout()


    quiver_opts = 'None'
    plot_amplitude(mesh, nC, active_cell_map,maxval,m, ax=ax[1,0])
    plot_amplitude(mesh, nC, active_cell_map,maxval,m, ax=ax[1,1], normal="Z", ind=zind)
    plt.tight_layout()

    #save figure and save with name of data file    
    plt.savefig(path+"/"+file[:-4]+".png")
    plt.close()


    fig, ax = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={'width_ratios': [2, 1]})
    plot_vector_model(mesh, nC, active_cell_map,maxval,m, ax=ax[0])
    plot_vector_model(mesh, nC, active_cell_map,maxval,m, ax=ax[1], normal="Z", ind=zind)
    plt.tight_layout()
    plt.savefig(path+"/"+file[:-4]+"_tot.png",bbox_inches='tight')
    plt.close()



# %%
