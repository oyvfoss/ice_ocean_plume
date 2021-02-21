### ICE_OCEAN_PLUME ###

---

STATUS 21.02 2021: 
Some minor tweaks of the output - core functionality seems good, but 
proper validation remains. 

STATUS 26.06 2020: 
Seems to work nicely, but yet to do extensive testing/validation.

---

Integrating the buoyant plume equations in the along-ice-face coordinate.

Returns stady-state plume properties (width, upwelling velocity, 
temperature, salinity etc) as a function of depth, as well as a 
number of other variables.

---

Inspired by (but not equivalent to!) previous Matlab application by Adrian
Jenkins and later modifications by Dustin Carroll.

---

Dependencies: Python 3 with scipy and numpy.

Likely not compatible with Python 2 - development and testing was done
in Python 3.7.3.

---

See test_ioplume_simple.py for an example of use.

(Additional dependencies: matplotlib)

---

Simple example:
===============

    P = plume.plume(gl_dep, volfl0, T_a, S_a, dep_a, theta = theta)
    P.set_params()
    P.solve(plume_type = 'line')


Inputs:
=======

    gl_dep:   Plume start depth (e.g. grounding line, ice shelf bottom) [m]
    volfl0    Initial volume flux 
                 Line plume: (Q / along-glacier width) [m2/s]
                 Cone plume: (Q) [m3/s]

    T_a:      Ambient water profile: in-situ temperature [C] 
    S_a:      Ambient water profile: practical salinity [psu]
    dep_a:    Ambient water profile: depth [m]
    theta:    Angle between glacier face and horizontal [deg] 
              (90 for vertical face)
    
    plume_type: 'line' (line plume) or
                'cone' (half-cone)


Dependent variables:
====================

(output plume variables have the suffix "_pl")

D: Plume width (line plume)  [m]
   Plume radius (cone plume) [m]
U: Plume upward velocity [m s⁻¹]
S: Plume (practical) salinity [psi]
T: Plume (in-situ) temperature [C]

In the integration, these are combined into a state variable:

Y: [DU, DU², DUT, DUS]


Independent variable
====================

x: Along-glacier coordinate upward from the grounding line [m]



Other key variables
===================

dep:    Depth coordinate [m]

θ:      Angle between ice face and horizontal  
        (90 for vertical face) [deg]

E:      Entrainment rate of ambient water [m s-1]

M:      Melt rate of ice face [m s-1]

Tb, Sb: Temperature and salinity of water at the plume-ice 
        interface

Ta, Sa: Temperature and salinity of ambient water in the 
        ocean surrounding the plume 

dRho:   Scaled density difference between the plume water 
        (T, S) and the ambient water (Ta, Sa): 
        (rho_plume - rho_amb) / rho_ref  [unitless]

GT, GS: Turbulent exchange coefficients of temperature and salinity   
        between the plume and the plume-ice interface

(See bottom for a list of constants)


Governing equations:
====================

Plume equations (Morton, 1956 and others) modified to include
ice-ocean interaction (Jenkins, 1991).

dY/dx = f(Y, x) 

For line plume (plume_type = 'line')

    (1) d(DU)/dx = E + M
    (2) d(DU²)/dx = D*dRho*g*sin(θ) - k*U²
    (3) d(DUT)/dx = E*Ta + M*Tb - GT*U*(T-Tb)
    (3) d(DUS)/dx = E*Sa + M*Sb - GS*U*(S-Sb)

For half-cone plume (plume_type = 'cone')

    (1) d(DU)/dx = (πD)*E + (2D)*M
    (2) d(DU²)/dx = (πD²/2)*dRho*g*sin(θ) - (2D)*k*U²
    (3) d(DUT)/dx = (πD)*E*Ta + (2D)*M*Tb - (2D)*GT*U*(T-Tb)
    (3) d(DUS)/dx = (πD)*E*Sa + (2D)*M*Sb - (2D)*GS*U*(S-Sb)


Initial conditions:
===================

The integration is initialized with the volume flux (volfl0)
specified by the user.

*Note: For a line plume, volfl0 is interpreted as a volume 
flux per unit width [m² s⁻¹], while for a cone plume it
is interpreted as a total volume flux [m² s⁻¹]

Initial values for D and U are then set by assuming 
no initial momentum flux (LHS of Equation (2) is zero).

For no initial discharge flux (NOT IMPLEMENTED YET!)


Boundary conditions
===================

Ambient water profiles (Ta, Sa) are necessary boundary conditions. 
During integration, these profiles are used to determine plume
buoyancy, and the properties of  water entrained in the plume.

Ice temperature and salinity (Ti, Si) are treated as constant 
parameters.


Parametrizations:
=================

Entrainment parameterization
----------------------------

  E = e0*U 

Turbulent exchange parameterization
----------------------------------

  GT = U*GamT
  GS = U*GamS

Melt parameterization
---------------------

  (a) Tb = FPa*Sb + FPb + FPc*dep
  (b) M*(Sb-Si) = GS*(S-Sb)
  (c) M*(L + ci*(Tb-Ti)) = GT*(S-Sb)


Additional parameters:
======================

Si: Ice salinity [psu] - typical value: 0
Ti: Ice temperature [C] - typical value: [-20 to -2]

e0: Entrainment coefficient [] - typical values [0.03 to 0.2]

g: Graviational acceleration [m s-2] - typical value: 9.81 
k: Drag coefficient [] - typical value: 2.5e-3

GamT: Thermal Stanton no. (transfer coeff.) [] - typical value: 1.1e-3
GamS: Haline Stanton no. (transfer coeff.) [] - typical value: 3.1e-5

FPa, FPb, FPc: Coefficients in freezing point equation - Tf(S, zdep)

L: Latent heat of fusion for ice [J kg-1] - typical value: 3.35e5
ci: Heat capacity, glacial ice [J kg-1 K-1] - typical value: 2009
c: Heat capacity, seawater [J kg-1 K-1] - typical value: 3974

BT: Haline contraction coeff [K-1] - typical value: 3.87e-5
BS: Haline contraction coeff [] - typical value: 7.86e-4


References:
===========

Jenkins, A. (1991). A one‐dimensional model of ice shelf‐ocean interaction. 
Journal of Geophysical Research: Oceans, 96(C11), 20671-20677.

Morton, B. R., Taylor, G. I., & Turner, J. S. (1956). Turbulent gravitational 
convection from maintained and instantaneous sources. Proceedings of the Royal 
Society of London. Series A. Mathematical and Physical Sciences, 234(1196), 
1-23.