import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..utils.plot import size
from .module import RPR
from .utils.logging import log, write
from .utils.target import points

R = 3000 * np.sqrt(3) / 6
r = 1000 * np.sqrt(3) / 6

main_rpr = RPR(
    ground_joints=R,
    platform_joints=r,
    Lmin=1000,
    Lmax=3000,
    name="RPR_main",
)

rprs: list[RPR] = [main_rpr]

# for i in range(10):
#     for joint in range(2):
#         ground_joints=main_rpr.ground.joints
#         for axis in ['x', 'y', 'xy']:
#             ground_joints[joint]=modify_vector(ground_joints[joint], offset_point, axis)
#             rpr = RPR(
#                     ground_joints=ground_joints,
#                     platform_joints=main_rpr.platform.joints,
#                     Lmin=main_rpr.Lmin, Lmax=main_rpr.Lmax,
#                     name=f"RPR_{i}_ground_{axis}",
#                 )
#             rprs.append(rpr)

# for i in range(10):
#     for joint in range(2):
#         platform_joints=main_rpr.platform.joints
#         for axis in ['x', 'y', 'xy']:
#             platform_joints[joint]=modify_vector(platform_joints[joint], offset_point, axis)
#             rpr = RPR(
#                     ground_joints=main_rpr.ground.joints,
#                     platform_joints=platform_joints,
#                     Lmin=main_rpr.Lmin, Lmax=main_rpr.Lmax,
#                     name=f"RPR_{i}_platform_{axis}",
#                 )
#             rprs.append(rpr)

xyz = [
    (coord, angle)
    for coord, angle in points(radius=500, limit=[10, 170], R=R, r=r, n=5000)
]


for i, rpr in tqdm(enumerate(rprs), total=len(rprs)):
    for idx, (coord, angle) in enumerate(xyz):
        rpr.move(
            coord=coord,
            angle=angle,
        )
        data = log(rpr)  # , ref=main_rpr)
        write(data, filename=f"src/rpr/data/test_{R}_{r}_{1000}_{10000}.csv")

        if idx % 100 == 0:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=size(np.array([10, 10])))
            rpr.plot(axis=ax)

            ax.margins(0.25)
            plt.show(block=False)
            plt.pause(0.1)
            time.sleep(1)
            plt.close()
