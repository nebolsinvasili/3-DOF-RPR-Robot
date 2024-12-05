import math

import numpy as np

"""
class Platform:
     def __init__(self, u) -> None:
        self.u = u


class Leg:
     def __init__(self, name, a):
        self.name = name
        self.a = a


     def print_coord_a(self):
         print(f"Точка крепления штанги {self.name}: {self.a}")
     
leg_1 = Leg(name='1', a=(0, 100, 0))
leg_2 = Leg(name='2', a=(math.sqrt(3)/2 * 100, -50, 0))
leg_3 = Leg(name='3', a=(-math.sqrt(3)/2 * 100, -50, 0))
print(leg_1.print_coord_a())
print(leg_2.print_coord_a())
print(leg_3.print_coord_a())

class RPR_3(Leg, Platform):
     def __init__(self) -> None:
        pass
"""
gr_TO_rad = lambda a: math.pi / 180 * a

R = 100 # Радиус точек крепления манипулятора к поверхности
r = 25 # Радиус платформы

Lmin, Lmax = 10, 190

xq = math.sqrt(3) * R / 2 # При R = 100: xq = 86.60254037844386
yq = R / 2 # При R = 100: yq = 50.0

A_1 = {'A1_x': 0,
       'A1_y': R,
       'A1_z': 0,
}

A_2 = {'A2_x': xq,
       'A2_y': -yq,
       'A2_z': 0,
}

A_3 = {'A3_x': -xq,
       'A3_y': -yq,
       'A3_z': 0,
}

A=[A_1, A_2, A_3]

j = {'j_1': math.pi / 2, 
     'j_2': - math.pi / 6, 
     'j_3': - 5 * math.pi / 6
     }

l3 = {'l3_1': 25, 
      'l3_2': 25, 
      'l3_3': 25
      }

x, y, fi = 0, 0, gr_TO_rad(90) #  Координаты TCP (англ. Tool Center Point — центральная точка инструмента)


def L(A, r, j, x, y, fi):
    return np.sqrt(np.power((x + r * np.cos(fi + j) - A[0]), 2) + np.power((y + r * np.sin(fi + j) - A[1]), 2))

Ls = {}
Ls.update({f'Ld_{idx+1}': L(A=a, r=r, j=j, x=x, y=y, fi=fi) for idx, (a, r, j) in enumerate(zip([list(A[0].values()), list(A[1].values()), list(A[2].values())], 
                                                                                                                 l3.values(), 
                                                                                                                 j.values()))})
print(Ls)
