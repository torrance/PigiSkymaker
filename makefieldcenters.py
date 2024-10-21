from casacore.tables import table
import numpy as np

# mset = table("/home/torrance/MWA/A2877/CHAN121/1214949848.ms")
# ra0, dec0 = mset.FIELD.getcell("PHASE_DIR", 0)[0]
ra0, dec0 = np.deg2rad(13), np.deg2rad(-47)

def lmtoradec(l, m, ra0, dec0):
    n = np.sqrt(1 - l**2 - m**2)
    ra = ra0 - np.arctan2(l, n * np.cos(dec0) - m * np.sin(dec0))
    dec = np.arcsin(m * np.cos(dec0) + n * np.sin(dec0))
    return ra, dec

scale = np.sin(np.deg2rad(12 / 3600))

lpx, mpx = -6000 - 2500, 0
ra, dec = np.rad2deg(lmtoradec(lpx * scale, mpx * scale, ra0, dec0))
print("[[image.fields]]")
print("width = 5000")
print("height = 8000")
print(f"projectioncenter = {{ra={ra}, dec={dec}}}")
print()

lpx, mpx = 0, 6000 + 2500
ra, dec = np.rad2deg(lmtoradec(lpx * scale, mpx * scale, ra0, dec0))
print("[[image.fields]]")
print("width = 8000")
print("height = 5000")
print(f"projectioncenter = {{ra={ra}, dec={dec}}}")
print()

lpx, mpx = 0, -6000 - 1500
ra, dec = np.rad2deg(lmtoradec(lpx * scale, mpx * scale, ra0, dec0))
print("[[image.fields]]")
print("width = 8000")
print("height = 3000")
print(f"projectioncenter = {{ra={ra}, dec={dec}}}")
print()

lpx, mpx = 6000 + 1500, 0
ra, dec = np.rad2deg(lmtoradec(lpx * scale, mpx * scale, ra0, dec0))
print("[[image.fields]]")
print("width = 3000")
print("height = 8000")
print(f"projectioncenter = {{ra={ra}, dec={dec}}}")
print()

# for lpx in [-5000, 5000]:
#     for mpx in [-2000, 2000]:
#         ra, dec = np.rad2deg(lmtoradec(lpx * scale, mpx * scale, ra0, dec0))
#         print("[[image.fields]]")
#         print("width = 4000")
#         print("height = 4000")
#         print(f"projectioncenter = {{ra={ra}, dec={dec}}}")
#         print()
# 
# for lpx in [-2000, 2000]:
#     for mpx in [-5000, 5000]:
#         ra, dec = np.rad2deg(lmtoradec(lpx * scale, mpx * scale, ra0, dec0))
#         print("[[image.fields]]")
#         print("width = 4000")
#         print("height = 4000")
#         print(f"projectioncenter = {{ra={ra}, dec={dec}}}")
#         print()
