#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 00:04:35 2023

@author: zhang
@author: kcmf2
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from summit import *


class MixedBenchmark(Experiment):
    """Kinetic model benchmark with mixed categorical and continuous variables"""

    def __init__(self, noise_level: float = 0):
        self.mixer_df = DataSet(
            [[0.5, 90], [1, 90], [0.5, 120], [1, 120]],
            index=["T_small", "T_big", "Y_small", "Y_big"],
            columns=["size", "angle"],
        )
        domain = self.setup_domain(self.mixer_df)
        # self.noise_level = noise_level
        super().__init__(domain)

    @staticmethod
    def setup_domain(mixer_df: DataSet):
        domain = Domain()

        # Decision variables
        domain += ContinuousVariable(
            name="equiv", description="equiv", bounds=[0.1, 2.0]
        )
        domain += ContinuousVariable(
            name="flowrate", description="flowrate", bounds=[0.1, 6.0]
        )
        domain += CategoricalVariable(
            name="elec",
            description="Electrophile",
            levels=["Acetic_chloride", "Acetic_anhydride"],
        )
        domain += CategoricalVariable(
            name="solv",
            description="Solvent",
            levels=["THF", "EtOAc", "MeCN", "Toluene"],
        )

        domain += CategoricalVariable(
            name="mixer", description="Mixer_geometry_descriptors", descriptors=mixer_df
        )

        # Objectives
        domain += ContinuousVariable(
            name="sty",
            description="Space Time Yield",
            bounds=[0, 100],
            is_objective=True,
            maximize=True,
        )
        domain += ContinuousVariable(
            name="e_factor",
            description="E Factor",
            bounds=[0, 10],
            is_objective=True,
            maximize=False,
        )

        return domain

    def _run(self, conditions: DataSet, plot: bool = False, **kwargs) -> DataSet:
        # Get selected conditions
        equiv = float(conditions["equiv"])
        flowrate = float(conditions["flowrate"])
        elec = str(conditions["elec"].values[0])
        solv = str(conditions["solv"].values[0])
        mixer = str(conditions["mixer"].values[0])
        mixer_diameter = float(self.mixer_df.loc[mixer, "size"].values[0])
        mixer_angle = float(self.mixer_df.loc[mixer, "angle"].values[0])

        c0, v, phi, k0 = self.get_parameters(
            equiv, flowrate, elec, solv, mixer_diameter, mixer_angle
        )
        tau, cA, cB, cC, cD = self.calculate(c0, k0, phi, v)
        if plot:
            self.plot(tau, cA, cB, cC, cD)
        yield_pred = round(100 * (cC[-1]) / 0.3, 2)
        sty, e_factor = self.calculate_obj(yield_pred, flowrate, equiv, elec)
        conditions["sty", "DATA"] = sty
        conditions["e_factor", "DATA"] = e_factor
        return conditions, {}

    @staticmethod
    def get_parameters(equiv, flowrate, elec, solv, mixer_dia, mixer_angle):
        v = 2e-3 * flowrate / 60
        c0 = np.array([equiv * 0.30, 0.30, 0, 0])

        if elec == "Acetic_anhydride":
            k0 = np.array([40, 0.2])
        else:
            k0 = np.array([30, 0.3])

        if solv == "Toluene":
            phi = 0.35
            k0[0] = 25
        elif solv == "EtOAc":
            phi = 0.35
            k0[0] = 60
        elif solv == "MeCN":
            if mixer_angle > 90:
                phi = 0.09 * flowrate - (1 / mixer_dia) * 0.05
                k0[0] = 40
            else:
                phi = 0.09 * flowrate + (1 / mixer_dia) * 0.06
                k0[0] = 40
        elif solv == "THF":
            if elec == "Acetic_anhydride" and mixer_angle > 90:
                phi = 0.19 * flowrate - (1 / mixer_dia) * 0.06
                k0[0] = 40

            elif elec == "Acetic_anhydride" and mixer_angle <= 90:
                phi = 0.19 * flowrate + (1 / mixer_dia) * 0.05
                k0[0] = 40
            elif elec == "Acetyl_chloride" and mixer_angle > 90:
                phi = 0.05 * flowrate - (1 / mixer_dia) * 0.05
                k0[0] = 40
            else:
                phi = 0.05 * flowrate + (1 / mixer_dia) * 0.02
                k0[0] = 40
        else:
            raise ValueError(f"Unknown solvent: {solv}")

        return c0, v, phi, k0

    @staticmethod
    def calculate(c0, k0, phi, v):
        # Parameters
        Nu = np.array([[-1, -1], [-1, 0], [1, 0], [0, 1]])  # Reaction stoichimetry
        Vr = 0.5 * 1e-3  # Reactor volume, L
        Nocomp = 4
        Noreac = 2
        order = np.array([[1, 1], [1, 0], [0, 0], [0, 0]])  # Reaction order

        def f(tau, c):
            dcdtau = np.zeros(Nocomp)
            Rate = np.zeros(Noreac)

            for i in range(0, Noreac):
                Rate[i] = k0[i] * np.prod((c * phi) ** order[:, i])  # mixing index phi

            for i in range(0, Nocomp):
                dcdtau[i] = np.sum(Rate * Nu[i, :])

            return np.array(dcdtau)

        # Integrate
        tau_span = np.array([0, Vr / v])
        spaces = np.linspace(tau_span[0], tau_span[1], 100)
        soln = solve_ivp(f, tau_span, c0, t_eval=spaces)

        # Extract results
        tau = soln.t
        cA = soln.y[0]
        cB = soln.y[1]
        cC = soln.y[2]
        cD = soln.y[3]

        return tau, cA, cB, cC, cD

    @staticmethod
    def plot(tau, cA, cB, cC, cD):
        plt.figure()
        plt.plot(tau, cA, "-", label="cA")
        plt.plot(tau, cB, "-", label="cB")
        plt.plot(tau, cC, "-", label="cC")
        plt.plot(tau, cD, "-", label="cD")

        plt.xlabel("res_time/s")
        plt.ylabel("conc/(mol/L)")
        plt.legend()
        plt.show()

    @staticmethod
    def calculate_obj(yield_pred, flowrate, equiv, elec):
        # calculate STY and E_factor
        Mw_product = 149.19
        Mw_anhyride = 102.09
        Mw_chloride = 78.5
        Mw_NaOH = 40

        res_time = 0.5 / (flowrate * 2 / 60)
        STY = 0.3 * Mw_product * (yield_pred / 100) / (0.5 * res_time)

        if elec == "Acetic_anhydride":
            waste = (
                107.15
                + equiv * (Mw_anhyride + Mw_NaOH)
                - Mw_product * (yield_pred / 100)
            )
        else:
            waste = (
                107.15
                + equiv * (Mw_chloride + Mw_NaOH)
                - Mw_product * (yield_pred / 100)
            )

        E_factor = waste / (Mw_product * (yield_pred / 100))

        return STY, E_factor


def test_benchmark():
    exp = MixedBenchmark()
    conditions = pd.DataFrame(
        [
            {
                "equiv": 1.4,
                "flowrate": 6,
                "elec": "Acetyl_chloride",
                "solv": "THF",
                "mixer": "T_small",
            }
        ]
    )
    conditions = DataSet.from_df(conditions)
    results = exp.run_experiments(conditions, plot=True)
    print(results)


if __name__ == "__main__":
    test_benchmark()
