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
    def __init__(self):
        domain = self.setup_domain()
        super().__init__(domain)

    def setup_domain(self):
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

        # Objectives
        domain += ContinuousVariable(
            name="yld",
            description="yield",
            bounds=[0, 100],
            is_objective=True,
            maximize=True,
        )
        return domain

    def _run(self, conditions: DataSet, plot: bool = False, **kwargs) -> DataSet:
        equiv = float(conditions["equiv"])
        flowrate = float(conditions["flowrate"])
        elec = str(conditions["elec"].values[0])
        solv = str(conditions["solv"].values[0])
        c0, v, phi, k0 = self.get_parameters(equiv, flowrate, elec, solv)
        tau, cA, cB, cC, cD = self.calculate(c0, k0, phi, v)
        if plot:
            self.plot(tau, cA, cB, cC, cD)
        yield_pred = round(100 * (cC[-1]) / 0.3, 2)
        conditions["yld", "DATA"] = yield_pred
        return conditions, {}

    @staticmethod
    def get_parameters(equiv, flowrate, elec, solv):
        v = 2e-3 * flowrate / 60
        c0 = np.array([equiv * 0.30, 0.30, 0, 0])

        if elec == "Acetic_anhydride":
            k0 = np.array([40, 0.2])
        else:
            k0 = np.array([30, 0.3])

        if solv == "Toluene":
            phi = 0.35
            k0[0] = 40
        elif solv == "EtOAc":
            phi = 0.35
            k0[0] = 100
        elif solv == "MeCN":
            phi = 0.09 * flowrate + 0.02
            k0[0] = k0[0] * 1
        elif solv == "THF":
            if elec == "Acetic_anhydride":
                phi = 0.19 * flowrate - 0.11
                k0[0] = k0[0] * 1
            else:
                phi = 0.05 * flowrate + 0.06
                k0[0] = k0[0] * 1
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


def test_benchmark():
    exp = MixedBenchmark()
    conditions = pd.DataFrame(
        [{"equiv": 1.4, "flowrate": 6, "elec": "Acetic_chloride", "solv": "THF"}]
    )
    conditions = DataSet.from_df(conditions)
    results = exp.run_experiments(conditions, plot=True)
    print(results)


if __name__ == "__main__":
    test_benchmark()
