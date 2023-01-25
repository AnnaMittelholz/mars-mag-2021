# PLotting functions to be used in helicopter scenarios
# L. Heagy and A. Mittelholz 

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


def full_mesh_magnetization(model, active_cell_map, nC):
    return np.vstack([active_cell_map * model.reshape(nC, 3, order="F")[:, i] for i in range(3)]).T


def plot_vector_model(
    mesh, nC, active_cell_map, maxval, model, ax=None, quiver_opts=None, normal="Y", xlim=None, ylim=None, ind=None, plot_data=True, plot_grid=False
):
    if ax is None: 
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    magnetization = full_mesh_magnetization(model, active_cell_map, nC)
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
        mesh, nC, active_cell_map,maxval, model, ax=None, quiver_opts=None, normal="Y", xlim=None, ylim=None, ind=None, plot_data=True, plot_grid=False
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
                
        magnetization = full_mesh_magnetization(model, active_cell_map, nC)
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

def plot_target_outline(ax, normal="Y", plot_opts={"color":"C3", "lw":3}):
    if normal.upper() == "X": 
        x_target = np.hstack([xyz_pts[:, 1], xyz_pts[::-1, 1], xyz_pts[0, 1]])
        y_target = np.hstack([xyz_pts[0, 2], xyz_pts[:, 2], xyz_pts[::-1, 2]])
    elif normal.upper() == "Y": 
        x_target = np.hstack([xyz_pts[:, 0], xyz_pts[::-1, 0], xyz_pts[0, 0]])
        y_target = np.hstack([xyz_pts[0, 2], xyz_pts[:, 2], xyz_pts[::-1, 2]])
    elif normal.upper() == "Z":
        x_target = np.hstack([xyz_pts[:, 0], xyz_pts[::-1, 0], xyz_pts[0, 0]])
        y_target = np.hstack([xyz_pts[0, 1], xyz_pts[:, 1], xyz_pts[::-1, 1]])
    
    ax.plot(x_target, y_target, **plot_opts)


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

