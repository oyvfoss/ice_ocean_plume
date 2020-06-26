'''
IOPLUME.PY

Integrating the plume equations in the along-ice-face coordinate *x*.

---

Copyright (c) 2020 Norwegian Polar institute under the MIT-License.

Written by Øyvind Lundesgaard (NPI).

Inspired by (but not analogous to!) previous Matlab application by Adrian
Jenkins and later modifications  by Dustin Carroll.
---


Example:
========

    P = plume.plume(gl_dep, volfl0, T_a, S_a, dep_a, theta = theta)
    P.set_params()
    P.solve(plume_type = 'line')

Inputs:
=======

    gl_dep:   Grounding line depth [m]
    volfl0    Initial volume flux 
                 Line plume: (Q / along-glacier width) [m2/s]
                 Cone plume: (Q) [m3/s]

    T_a:      Ambient water profile: in-situ temperature [C] 
    S_a:      Ambient water profile: practical salinity [psu]
    dep_a:    Ambient water profile: depth [m]
    theta:    Angle between glacier face and horizontal [deg] 
              (90 for vertical face)
    
    plume_type: 'line' (axisymmetric line plume) or
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


Governing equations:
====================

Plume equations (Morton, 1956 and others) modified to include
ice-ocean interaction (Jenkins, 1991).

dY/dx = f(Y, x) 

For axisymmetric line plume (plume_type = 'line')

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

'''

import numpy as np
from scipy.integrate import solve_ivp, simps
from scipy.interpolate import interp1d

class plume():

    '''
    Initializing:
    --------------

    gl_dep:       Grounding line depth [m]
    volfl0:         Discharge flux (Q/D) [m2/s]
    T_a:          Ambient water profile: in-situ temperature [C] 
    S_a:          Ambient water profile: practical salinity [psu]
    dep_a:        Ambient water profile: depth [m]
    theta:        Angle between glacier face and horizontal [deg] 
    plume_type:   'line' or 'cone'

    Example (minimal, with default options): 
    ----------------------------------------
        P = plume.plume(gl_dep, volfl0, T_a, S_a, dep_a)
        P.set_params()
        P.solve()

    Example (with some tweaks): 
    ---------------------------

        P = plume.plume(gl_dep, volfl0, T_a, S_a, dep_a, theta = 80, 
                    plume_type = 'cone')
        P.set_params(Si = -10, e0 = 0.036)
        P.solve(Sinit = 1, Tfreeze = True)
    '''

###############################################################################
######  MODEL SETUP (Initializing the model case and setting parameters)  #####
###############################################################################

    def __init__(self, gl_dep, volfl0, T_a, S_a, dep_a, theta = 90, 
                 plume_type = 'line',):
        '''
        Initializing a new plume object.
        '''
        self.gldep = gl_dep
        self.volfl0 = volfl0
        self.Ta_o = T_a
        self.Sa_o = S_a
        self.depa_o = dep_a
        self.theta = theta
        self.plume_type = plume_type

        if gl_dep > dep_a.max():
            raise Exception(
                'Grounding line depth *gl_dep* (%.1f m) must not'%gl_dep
                + ' exceed the deepest point on the ambient profile'
                + ' *dep_a* (%.1f m).'%dep_a.max())

        if not 90 >= self.theta > 0:
            raise Exception(
                'Invalid theta value: %.1f degrees. '%theta
                +' (Theta should be <0, 90] degrees.)')

        
        self.sinth = np.sin(self.theta*np.pi/180) # Calculate sin(theta)

        # Interpolants for ambient temp/sal
        self.T_ambient_ip = interp1d(self.depa_o, self.Ta_o, kind = 'linear')
        self.S_ambient_ip = interp1d(self.depa_o, self.Sa_o, kind = 'linear')

###############################################################################

    def set_params(self, **change_params):
        '''
        Set some constant parameters.

        Any parameter can be modified when calling set_params(), e.g.:

            P.set_params(e0=0.036, Ti = -10)

        Parameters
        ----------
        e0: Entrainment coefficient - default: {0.1}

        Si: Ice salinity - default: {0} 
        Ti: Ice temperature - default: {-2} C

        k: Quadratic drag coefficient - default: {2.5e-3}
        g: Gravitational acceleration - default: {9.81} m s⁻² 

        GamT: Thermal Stanton no. (transfer coeff.) - default: {1.1e-3}
        GamS: Haline Stanton no. (transfer coeff.)- default: {3.1e-5}

        L: Latent heat of fusion for ice - default: {3.35e5} J kg⁻¹ 

        ci: Heat capacity, glacial ice - default: {2009} J kg⁻¹ K⁻¹
        c: Heat capacity, seawater- default: {3974} J kg⁻¹ K⁻¹

        BT: Haline contraction coeff - default: {3.87e-5} K⁻¹
        BS: Haline contraction coeff - default: {7.86e-4}

        FPa, FPb, FPc: Coefficients in freezing point equation -
                       default: {-0.0573, 0.0832, 7.61e-4}

        u_eps: Minimum initial velocity - default: {1e-8} m s⁻¹

        '''
        self.e0 = 0.1 # Entrainment coefficient

        self.Si = 0 # Ice salinity
        self.Ti = -2 # Ice temperature

        self.k = 2.5e-3 # Drag coefficient
        self.g = 9.81 # Grav accel

        # Stanton numbers 
        self.GamT = 1.1e-3
        self.GamS = 3.1e-5

        # Latent heat of fusion for ice
        self.L = 3.35e5
        
        # Heat capacities
        self.ci = 2009.0 # Ice
        self.c = 3974.0 # Water
 
        # Thermal and haline contraction coeffictients
        self.BT = 3.87e-5
        self.BS = 7.86e-4

        # Coefficients in the freezing point of seawater.
        self.FPa = -0.0573
        self.FPb = 0.0832
        self.FPc = 7.61e-4

        # Minimum velocity
        self.u_eps = 1e-8 

        # Optional: Set values
        for key, value in change_params.items():
            setattr(self, key, value)


###############################################################################
########  MODEL EXECUTION (Running the model and preparing the output) ########
###############################################################################

    def solve(self, Tinit = 0, Sinit = 0, melt_on = True, method = 'RK45', 
              max_step = 0.5, manual_step = False):

        '''
        Solve the plume equations using the 
        scipy.integrate.solve_ivp solver.
        
        plume_type: ['line' or 'cone'] chooses which model 
                     formulation to use.

        Tinit, Sinit: Initial plume temperature and salinity.

        melt_on: Include ice-ocean interactions.

        method: Integration method (see documentation for 
                scipy.integrate.solve_ivp). 
                Default: 'RK45' (Explicit Runge-Kutta method of 
                order 5(4)).

        max_step: Maximum integration step size.

        manual_step: if toggled on (True), a diagnostic string is
                     printed at each integration step. 
        '''

        # Get initial conditions
        Y_init = self.get_Yinit(Tinit = Tinit, Sinit = Sinit,)

        # Making a function wrapper so we can feed the function to the 
        # solver with arguments
        dYdt_function_wrapper = lambda x, y: self.get_dYdt( 
            x, y, manual_step = manual_step, melt_on = melt_on)

        # Setting "event functions" that are fed to the solver.
        # This allows us to track when we reach neutral and maximum depth. 
        # Functions are defined further down, creating wrappers here.
        event_top_plume_wrapper =lambda x, y: self.event_top_plume(y)  
        event_neutral_wrapper=lambda x, y: self.event_neutral(x, y)  
        # Cause integration to terminate when the top is reached.
        event_top_plume_wrapper.terminal = True 

        ### SOLVING ###
        SOL = solve_ivp(dYdt_function_wrapper, 
                        [0, self.gldep/self.sinth], Y_init, 
                        events = [event_top_plume_wrapper, event_neutral_wrapper], 
                        vectorized = True, method = method, 
                        dense_output = False, max_step = max_step)

        print('%s plume integration complete. (Success: %s)'%(
                    self.plume_type.upper(), SOL.success))

        # Extract variables from solution
        self.x_pl = SOL.t
        self.dep_pl = self.gldep - self.sinth*SOL.t
        self.D_pl = SOL.y[0]**2 / SOL.y[1]
        self.U_pl = SOL.y[1] / SOL.y[0]
        self.T_pl = SOL.y[2] / SOL.y[0]
        self.S_pl = SOL.y[3] / SOL.y[0]
        self.E_pl = self.U_pl*self.e0

        self.drho = self.get_density_diff(
            self.T_ambient_ip(self.dep_pl), self.T_pl, 
            self.S_ambient_ip(self.dep_pl), self.S_pl)

        # Extract minimum depth
        if SOL.status==0:   # SOL.status is 0 if x=self.gldep (dep=0) was 
                            # reached, i.e., if the plume reached the surface.
            self.min_dep = 0 
            self.surface_plume = True

        elif SOL.status==1: # SOL.status is 1 if a termination event occured, 
                            # i.e. if the plume reached a subsurface min depth.
            self.min_dep = self.gldep - self.sinth*SOL.t_events[0][0]
            self.surface_plume = False

        else:               # SOL.status is -1 if the integration failed.
            excpt_str = ('INTEGRATION FAILED. '
                         + 'Termination reason: %s'%SOL.message)
            raise Exception(excpt_str)

        # Extract neutral depth 
        if len(SOL.t_events[1]) == 1: # If the plume has a neutral depth
            self.neut_dep = self.gldep - self.sinth*SOL.t_events[1][0]
            self.has_neut_dep = True

        else:
            self.neut_dep = 0
            self.has_neut_dep = False

        # If the terminal point is U=0, D explodes here.
        # Therefore setting D(terminal) to nan if this is the case.


        # Extract and recompute various useful quantities/variables
        self.recompute_from_solution(SOL.y)
        self.compute_terminal_properties()
        self.compute_total_melt_entr()

###############################################################################

    def get_dYdt(self, x, Y, melt_on = True, manual_step = False,):
        '''
        Calculate the LHS of the differential equation set.

        Inputs: Depth (x) and state variable (Y).

        Output: LHS (dYdt).

        plume_type ['line' or 'cone'] chooses which model 
        formulation to use.

        If manual_step is toggled (True), a diagnostic string is
        printed at each step. 
        '''

        if np.isnan(Y).any():
            raise Exception('''
                Returned NaN at depth %.1f: 
                [DU, DU^2, DUT, DUS] = [%.1f, %.1f, %.1f, %.1f]
                '''%(x, *Y))

        # Read core variables from state variable
        D_ = Y[0]**2 / Y[1]
        U_ = Y[1] / Y[0]
        T_ = Y[2] / Y[0]
        S_ = Y[3] / Y[0]

        # Calculate depth         
        dep = self.gldep - self.sinth*x 

        # Calculate ice-ocean interface quantities
        M, Tb, Sb, GT, GS, Tf = self.get_melt(Y, dep, melt_on = melt_on)

        # Calculate entrainment rate
        E = self.e0 * U_

        # Grab ambient water properties
        Ta = self.T_ambient_ip(dep)
        Sa = self.S_ambient_ip(dep)

        # Calculate density difference between plume and ambient
        dRho = self.get_density_diff(Ta, T_, Sa, S_)

        # Calculate LHS of line plume equations
        if self.plume_type == 'line':
            dDU_dt = E + M
            dDUU_dt = D_*dRho*self.g - self.sinth*self.k*U_**2 
            dDUT_dt = E*Ta + M*Tb - GT*U_*(T_-Tb)
            dDUS_dt = E*Sa + M*Sb - GS*U_*(S_-Sb)

        elif self.plume_type == 'cone':
            dDU_dt = (np.pi*D_)*E + (2*D_)*M
            dDUU_dt = ((D_**2*np.pi/2)*dRho*self.g 
                      - self.sinth*(2*D_)*self.k*U_**2)
            dDUT_dt = (np.pi*D_)*E*Ta + (2*D_)*M*Tb - (2*D_)*GT*U_*(T_-Tb)
            dDUS_dt = (np.pi*D_)*E*Sa + (2*D_)*M*Sb - (2*D_)*GS*U_*(S_-Sb)

        else: 
            raise Exception("plume_type must be 'line' or 'cone' "
                            + '(failed with plume_type=%s)'%plume_type)

        dYdt = [dDU_dt, dDUU_dt, dDUT_dt, dDUS_dt]

        # Optional: Print the state vector at every step (for diagnostics):
        if manual_step: 
            stepstr = '''
                Ta: %.2f, Sa: %.2f, Tb: %.2f, Sb: %.2f,
                dep: %.1f, D: %.2e, U: %.2e, dRho: %.2e
                (press any key to continue..)
                '''(Ta, Sa, Tb, Sb, dep, D_, U_, dRho)
            dummy = input(manual_step_str) # Print diag string and 
                                           # wait for input to continue.

        return dYdt

###############################################################################

    def get_Yinit(self, Tfreeze = True, Tinit = 0, Sinit = 0):
        '''
        Calculate initial conditions for the plume.
    
        T and S can be specified. If Tfreeze = True, the temperature is 
        set to the pressure-dependent freezing point
        '''

        # Set the initial temperature to the freezing point
        if Tfreeze:
            Tinit = Sinit*self.FPa + self.FPb + self.FPc*self.gldep

        # Get ambient T, S and density difference at the grounding line
        Ta_init = self.T_ambient_ip(self.gldep)
        Sa_init = self.S_ambient_ip(self.gldep)
        drho_init = self.get_density_diff(Ta_init, Tinit, Sa_init, Sinit)

        # Calculate initial plume D and U by assuming no initial u
        # upward momentum flux (setting LHS of (2) to 0):

        if self.plume_type == 'line':        
            self.U_init = (drho_init*self.g*self.volfl0*self.sinth
                           /(self.e0+self.k))**(1/3) 
            self.D_init = self.volfl0/self.U_init
            
        if self.plume_type == 'cone':        
            self.U_init = (np.sqrt(np.pi*self.volfl0/8/self.k**2)
                           *(drho_init*self.g*self.sinth))**(2/5) 
            self.D_init = np.sqrt(self.volfl0/(np.pi*self.U_init))

        # Store initial temperature and salinity
        self.T_init = Tinit
        self.S_init = Sinit

        # Return initial state variable
        Yinit = [self.D_init*self.U_init, 
                 self.D_init*self.U_init**2, 
                 self.D_init*self.U_init*self.T_init, 
                 self.D_init*self.U_init*self.S_init]

        return Yinit

###############################################################################

    def recompute_from_solution(self, Y):
        '''
        Recompute along-plume properties (including melt and ambient variables) 
        from a complete plume solution.
        '''

        N = Y.shape[1] # Number of points in plume solution.

        # Collecting into (temporary) variable dictionary:
        VD_ = {}

        # Get depth
        VD_['dep'] = self.dep_pl

        # Get ambient water properties  
        VD_['Ta'] = self.T_ambient_ip(self.dep_pl)
        VD_['Sa'] = self.S_ambient_ip(self.dep_pl)

        # Get plume-ambient density difference (scaled and unscaled) 
        VD_['dRho'] = self.get_density_diff(VD_['Ta'], self.T_pl, 
                                                VD_['Sa'], self.S_pl)
        # Get melt parameters

        varkeys_melt = ['M', 'GT', 'GS', 'Tf', 'Tb', 'Sb']
        
        for varkey in varkeys_melt:
            VD_[varkey] = np.ma.zeros(N)
            VD_[varkey].mask = True

        for nn in np.arange(N): # Looping through the plume solution
            (VD_['M'][nn], VD_['Tb'][nn], VD_['Sb'][nn], VD_['GT'][nn], 
            VD_['GS'][nn], VD_['Tf'][nn]) = (
                    self.get_melt(Y[:, nn], self.dep_pl[nn]))

        # Save as attributes
        for varkey in varkeys_melt + ['dRho', 'dep', 'Ta', 'Sa']:
            setattr(self, varkey+'_pl', VD_[varkey])

        # Freezing point of ambient water
        self.Tf_a = self.FPa*self.Sa_pl + self.FPb + self.FPc*self.dep_pl

###############################################################################

    def compute_terminal_properties(self):
        '''
        Compute plume properties at minimum and neutral depth from a 
        complete plume solution.
        '''

        terminal_vars = ['D', 'U', 'T', 'S', 'M', 'Ta', 'Sa']

        # Read values at terminal depth
        for key in terminal_vars:
            keynm_pl, keynm_terminal = '%s_pl'%key, '%s_mindep'%key
            val_terminal = getattr(self, keynm_pl)[-1]
            setattr(self, keynm_terminal, val_terminal)

        # Read values at neutral depth
        if self.has_neut_dep:
            for key in terminal_vars:
                keynm_pl, keynm_neut = '%s_pl'%key, '%s_neut'%key
                # Interpolate variable onto depth of neutral buoyancy 
                val_neut = interp1d(self.dep_pl, 
                                    getattr(self, keynm_pl))(self.neut_dep)  
                setattr(self, keynm_neut, val_neut)


        # Read volume flux at terminal and neutral depth
        if self.plume_type == 'line':
            self.volfl_mindep = self.D_pl[-1]*self.U_pl[-1]
            self.volfl_neut = interp1d(self.dep_pl, 
                                       self.D_pl*self.U_pl)(self.neut_dep)
        elif self.plume_type == 'cone':
            self.volfl_mindep = np.pi/2*self.D_pl[-1]**2*self.U_pl[-1]
            self.volfl_neut = interp1d(self.dep_pl, 
                                       np.pi/2*self.D_pl**2
                                       *self.U_pl)(self.neut_dep)
        else:
            for key in terminal_keys +  ['DU']:
                setattr(D, '%s_neut'%key, False)


###############################################################################

    def compute_total_melt_entr(self):
        '''
        Compute total meltwater and ambient water entrained in the 
        plume from bottom to neutral and minimum depth.

        Also compute the partition meltwater / entrained ambient 
        water / initial volume flux in the plume at terminal depth
        (defined as either neutral depth or surface).

        For a line plume: 
            - Meltwater flux per along-glacier width [m²/s]  
            - Entrainment flux per along-glacier width [m²/s]  

        For a cone plume: 
            - Total meltwater flux [m³/s]  
            - Total entrainment flux [m³/s]  

        Numerical integration is done using scipy.integrate.simps.

        Run *after* recompute_from_solution() and 
        compute_terminal_properties().
        '''

        # Compute integrated melt/entrainment rate for the whole plume 
        # (up to minimum depth).

        if self.plume_type == 'line':
            self.melt_total = simps(self.M_pl[::-1][1:], self.dep_pl[::-1][1:])
            self.entr_total = simps(self.E_pl[::-1][1:], self.dep_pl[::-1][1:])

        elif self.plume_type == 'cone':
             self.melt_total = simps(2*self.D_pl[::-1][1:]*self.M_pl[::-1][1:], 
                                     self.dep_pl[::-1][1:])
             self.entr_total = simps(np.pi*0.5*self.D_pl[::-1][1:]**2
                                     *self.E_pl[::-1][1:], 
                                     self.dep_pl[::-1][1:])


        # Compute integrated melt/entrainment rate up to neutral depth.
        # If neutral depth is not reached: Integrating over the 
        # whole plume

        if self.has_neut_dep:
            #   (Index of last point before crossing neutral depth.)
            neut_ind = np.ma.where(self.dep_pl-self.neut_dep < 0)[0][0]
            #   (M, D and dep up to  - and including - the neutral depth.) 
            M_to_neut = np.append(self.M_pl[:neut_ind], self.M_neut)
            E_to_neut = np.append(self.E_pl[:neut_ind], self.M_neut)
            D_to_neut = np.append(self.D_pl[:neut_ind], self.D_neut)
            dep_to_neut = np.append(self.dep_pl[:neut_ind], self.neut_dep)

            if self.plume_type == 'line':
                self.melt_to_neutral = simps(M_to_neut[::-1], 
                                             dep_to_neut[::-1])
                self.entr_to_neutral = simps(E_to_neut[::-1], 
                                             dep_to_neut[::-1])
            if self.plume_type == 'cone':
                self.melt_to_neutral = simps(2*D_to_neut[::-1]
                                             *M_to_neut[::-1],
                                             dep_to_neut[::-1])
                self.entr_to_neutral = simps(np.pi*0.5*D_to_neut[::-1]**2
                                             *E_to_neut[::-1],
                                             dep_to_neut[::-1])
        else:
            self.melt_to_neutral = self.melt_total 
            self.entr_to_neutral = self.entr_total 


        # Get fraction meltwater / entrained ambient water / initial volume flux
        # in the plume at terminal depth (surface or neutral depth).

        # Computing from the initial and integrated fluxes. Small deviations
        # from terminal level volume flux may occur - if high precision is 
        # necessary: reduce the step size (max_step = 0.5 in solve()).
        
        if self.has_neut_dep:
            self.terminal_volflux = (self.melt_to_neutral + self.entr_to_neutral 
                                     + self.volfl0)
            self.terminal_frac_melt = self.melt_to_neutral/self.terminal_volflux 
            self.terminal_frac_entr = self.entr_to_neutral/self.terminal_volflux 
            self.terminal_frac_volfl0 = self.volfl0/self.terminal_volflux 



###############################################################################
#########  PARAMETERIZATIONS (computing melt and density difference)  #########
###############################################################################

    def get_melt(self, Y, dep, melt_on = True):
        ''' 
        Evaluating the 3-equation melt formulation.

        Returning M, Tb, Sb, GT, GS, Tf.

        If melt_on is off: melt and turbulent fluxes set to zero, Tb 
        and Sb (arbitrarily) set to 1e5.

        '''
        # Get U and S from state variable Y
        # (Set U to minimum value if necessary)
        T_ = Y[2] / Y[0]
        S_ = Y[3] / Y[0]
        U_ = max([Y[1] / Y[0], self.u_eps])
        
        # Calculate the freezing point of the plume water 
        Tf = self.FPa*S_ + self.FPb + self.FPc*dep
        # Calculate the freezing point for pure meltwater
        Tfi = self.FPa*self.Si + self.FPb + self.FPc*dep

        # Return nothing if melt_on is toggled off
        if melt_on == False:
            M, Tb, Sb, GT, GS = 0, 1e5, 1e5, 0, 0
            return M, Tb, Sb, GT, GS, Tf

        # Calculate turbulent exchange coefficients
        GT = U_*self.GamT*1
        GS = U_*self.GamS*1

        # Mbool is 1 when melting occurs
        Mbool = int(T_>Tf)

        # Coefficients of the quadratic equation for M
        Q1 = self.L +Mbool*(self.ci)*(Tfi-self.Ti)
        Q2 = GS*(self.L + Mbool*(self.ci)*(Tf-self.Ti)) + GT*self.c*(Tfi-T_)
        Q3 = self.c*GS*GT*(Tf-T_)

        # Calculate M
        M = -(Q2-np.sqrt((Q2**2)-4*Q1*Q3))/(2*Q1)

        # Calculate boundary T and S
        Tb = ((self.c*GT*T_ + self.ci*M*self.Ti - self.L*M) 
             / (GT*self.c + self.ci*M))
        Sb = ((Tb - self.FPb - self.FPc*dep)/self.FPa)

        return M, Tb, Sb, GT, GS, Tf

###############################################################################

    def get_density_diff(self, Ta, T, Sa, S, ):
        '''
        Get the scaled density difference between the plume (T, S) and 
        the ambient water (Ta, Sa).

        dRho = (rho_plume - rho_amb)/rho_reference
        '''

        dRho = self.BS*(Sa-S)-self.BT*(Ta-T)
        return dRho

###############################################################################
##  EVENT FUNCTIONS (telling solver when to register neutral/terminal depth) ##
###############################################################################

    def event_neutral(self, x, Y):
        '''
        Finds the neutral depth by finding where the density 
        difference is minimized.
        '''
        Ta = self.T_ambient_ip(self.gldep-x)
        Sa = self.S_ambient_ip(self.gldep-x)
        dRho = self.get_density_diff(Ta, Y[2]/Y[0], Sa, Y[3]/Y[0])
        return dRho

###############################################################################

    def event_top_plume(self, Y):
        '''
        Finds the top plume depth by determining where U is minimized. 

        Use a small threshold value 1e7 for numerical purposes.
        '''
        return Y[1]/Y[0]-1e-7

###############################################################################