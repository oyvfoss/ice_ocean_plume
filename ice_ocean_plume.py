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

    P = plume.plume(dep0, volfl0, T_a, S_a, dep_a, theta = theta)
    P.set_params()
    P.solve(plume_type = 'line')

Inputs:
=======

    dep0:   Grounding line depth [m]
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

For axisymmetric half-cone plume (plume_type = 'cone')

    (1) d(D²U)/dx = (2D)*E + (4D/π)*M
    (2) d(D²U²)/dx = D²*dRho*g*sin(θ) - (4D/π)*k*U²
    (3) d(D²UT)/dx = (2D)*E*Ta + (4D/π)*M*Tb - (4D/π)*GT*U*(T-Tb)
    (3) d(D²US)/dx = (2D)*E*Sa + (4D/π)*M*Sb - (4D/π)*GS*U*(S-Sb)


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

  E = e0*U*sin(θ)

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

    dep0:         Grounding line depth [m]
    volfl0:       Discharge flux (Q/D) [m2/s]
    T_a:          Ambient water profile: in-situ temperature [C] 
    S_a:          Ambient water profile: practical salinity [psu]
    dep_a:        Ambient water profile: depth [m]
    theta:        Angle between glacier face and horizontal [deg] 
    plume_type:   'line' or 'cone'
    
    T0, S0:       Initial plume temperature [C] and salinity [psu]. 
    T0freeze:     'True' sets T0 to T_f(S0). Overrides T0.
    T0melt:        'True' sets T0, S0 to mixture between ambient water
                  and ocean-driven melt. Overrides T0, S0.
    frac_melt:    if *T0melt* is activated: sets the amount of meltwater 
                  in the initial plume, from *frac_melt=0* (no 
                  meltwater) along the Gade line to *frac_melt=1* 
                  ("meltwater saturated ambient water").


    Example (minimal, with default options): 
    ----------------------------------------
        P = plume.plume(dep0, volfl0, T_a, S_a, dep_a)
        P.set_params()
        P.solve()

    Example (with some tweaks): 
    ---------------------------

        P = plume.plume(dep0, volfl0, T_a, S_a, dep_a, theta = 80, 
                    plume_type = 'cone')
        P.set_params(Si = -10, e0 = 0.036)
        P.solve(S0 = 1, T0freeze = True)
    '''



###############################################################################
######  MODEL SETUP (Initializing the model case and setting parameters)  #####
###############################################################################

    def __init__(self, dep0, volfl0, Ta, Sa, depa, theta = 90, 
                 plume_type = 'line', T0 = 0, S0 = 0, T0freeze = False, 
                 T0melt =False, frac_melt = 1):
        '''
        Initializing a new plume object.
        '''
        init_input = locals()

        for key in init_input.keys():
            setattr(self, key, init_input[key])

        if dep0 > depa.max():
            raise Exception(
                'Grounding line depth *dep0* (%.1f m) must not'%dep0
                + ' exceed the deepest point on the ambient profile'
                + ' *dep_a* (%.1f m).'%dep_a.max())

        if not 90 >= self.theta > 0:
            raise Exception(
                'Invalid theta value: %.1f degrees. '%theta
                +' (Theta should be <0, 90] degrees.)')

        
        self.sinth = np.sin(self.theta*np.pi/180) # Calculate sin(theta)

        # Interpolants for ambient temp/sal
        self.T_ambient_ip = interp1d(self.depa, self.Ta, kind = 'linear')
        self.S_ambient_ip = interp1d(self.depa, self.Sa, kind = 'linear')

        # Internal variable used to control explicit output when solving
        self.prompt_to_continue_ = True

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

    def solve(self, melt_on = True, method = 'RK45', 
              max_step = 0.5, manual_step = False):

        '''
        Solve the plume equations using the 
        scipy.integrate.solve_ivp solver.
        
        plume_type: ['line' or 'cone'] chooses which model 
                     formulation to use.

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
        Y_init = self.get_Yinit()

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
                        [0, self.dep0/self.sinth], Y_init, 
                        events = [event_top_plume_wrapper, 
                                  event_neutral_wrapper], 
                        vectorized = True, method = method, 
                        dense_output = False, max_step = max_step)

        print('%s plume integration complete. (Success: %s)'%(
                    self.plume_type.upper(), SOL.success))

        # Extract variables from solution
        self.x_pl = SOL.t
        self.dep_pl = self.dep0 - self.sinth*SOL.t
        self.D_pl = SOL.y[0]**2 / SOL.y[1]
        self.U_pl = SOL.y[1] / SOL.y[0]
        self.T_pl = SOL.y[2] / SOL.y[0]
        self.S_pl = SOL.y[3] / SOL.y[0]
        self.E_pl = self.U_pl*self.e0

        self.drho = self.get_density_diff(
            self.T_ambient_ip(self.dep_pl), self.T_pl, 
            self.S_ambient_ip(self.dep_pl), self.S_pl)

        # Extract minimum depth
        if SOL.status==0:   # SOL.status is 0 if x=self.dep0 (dep=0) was 
                            # reached, i.e., if the plume reached the surface.
            self.min_dep = 0 
            self.surface_plume = True

        elif SOL.status==1: # SOL.status is 1 if a termination event occured, 
                            # i.e. if the plume reached a subsurface min depth.
            self.min_dep = self.dep0 - self.sinth*SOL.t_events[0][0]
            self.surface_plume = False

        else:               # SOL.status is -1 if the integration failed.
            excpt_str = ('INTEGRATION FAILED. '
                         + 'Termination reason: %s'%SOL.message)
            raise Exception(excpt_str)

        # Extract neutral depth 
        if len(SOL.t_events[1]) == 1: # If the plume has a neutral depth
            self.neut_dep = self.dep0 - self.sinth*SOL.t_events[1][0]
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

        # Remove internal attribute
        delattr(self, 'prompt_to_continue_')


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
            if self.plume_type == 'line':
                Ystr = '[DU, DU^2, DUT, DUS]'
            elif self.plume_type == 'cone':
                Ystr = '[D^2U, D^2U^2, D^2UT, D^2US]'

            raise Exception('''
                Returned NaN at depth %.1f: 
                %s = [%.1f, %.1f, %.1f, %.1f]
                '''%(self.dep0 - self.sinth*x, Ystr, *Y))

        # Read core variables from state variable

        U_ = Y[1] / Y[0]
        T_ = Y[2] / Y[0]
        S_ = Y[3] / Y[0]

        if self.plume_type == 'line':
            D_ = Y[0]**2 / Y[1]

        elif self.plume_type == 'cone':
            try:
                D_ = np.sqrt(Y[0]**2 / np.abs(Y[1]))
            except:
                import pdb
                pdb.set_trace()

        else: 
            raise Exception("plume_type must be 'line' or 'cone' "
                            + '(failed with plume_type=%s)'%plume_type)

        # Calculate depth         
        dep = self.dep0 - self.sinth*x 

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
            dDU_dt = E*self.sinth + M
            dDUU_dt = D_*dRho*self.g*self.sinth - self.k*U_**2 
            dDUT_dt = E*Ta + M*Tb - GT*U_*(T_-Tb)
            dDUS_dt = E*Sa + M*Sb - GS*U_*(S_-Sb)
            
            dYdt = [dDU_dt, dDUU_dt, dDUT_dt, dDUS_dt]

        elif self.plume_type == 'cone':
            dDDU_dt = (2*D_)*E + (4*D_/np.pi)*M
            dDDUU_dt = ( (D_**2)*dRho*self.g*self.sinth
                       - (4*D_/np.pi)*self.k*U_**2 )
            dDDUT_dt = ( (2*D_)*E*Ta + (4*D_/np.pi)*M*Tb 
                        - (4*D_/np.pi)*GT*U_*(T_-Tb) )
            dDDUS_dt = ( (2*D_)*E*Sa + (4*D_/np.pi)*M*Sb  
                        - (2*D_/np.pi)*GS*U_*(S_-Sb) )

            dYdt = [dDDU_dt, dDDUU_dt, dDDUT_dt, dDDUS_dt]


        # Optional: Print the state vector at every step (for diagnostics):
        if manual_step: 
            stepstr = (
                'Ta: %.2f, Sa: %.2f, Tb: %.2f, Sb: %.2f, '%(Ta, Sa, Tb, Sb)
                +'dep: %.1f, D: %.2e, U: %.2e, dRho: %.2e'%(dep, D_, U_, dRho))

            contstr = ('(press any key to continue, or "r" to' 
                        + ' run without prompt..)')
            
            if self.prompt_to_continue_:
                # Print diag string and wait for input to continue:
                dummy = input(stepstr + contstr) 
            else:
                # Print diag string without prompting
                dummy = print(stepstr)         

            if dummy == 'r':
                self.prompt_to_continue_ = False

        return dYdt

###############################################################################

    def get_Yinit(self):
        '''
        Calculate initial conditions for the plume.
    
        Initial T and S (T0, S0) can be specified. 
        
        If T0freeze = True, initial temperature T0 is set to the pressure-
        dependent freezing point at salinity S0.
        
        If T0melt = True, the plume is initialized with a mixture of 
        ambient water and ocean-driven meltwater. The latter calculated
        from ambient temperature by moving along the "Gade line". The 
        mixture is given by the parameter *frac_melt*:

            frac_melt = 0 : Initial plume 100% ambient water
            frac_melt = 1 : Initial plume ocean-driven meltwater at the freezing 
                            point ("meltwater saturated ambient water") 
        
        '''

        if self.T0freeze and self.T0melt:
            raise Exception(
                'Error in get_Yinit: Options *T0freeze* and *T0melt* are in' 
                'conflict, and both cannot be set to True'
                )
       

        # Get ambient T, S at the initial plume depth
        Ta0 = self.T_ambient_ip(self.dep0)
        Sa0 = self.S_ambient_ip(self.dep0)

        # Set the initial temperature to the freezing point
        if self.T0freeze:
            self.T0 = self.S0*self.FPa + self.FPb + self.FPc*self.dep0

        if self.T0melt:
            self.T0, self.S0 = self.get_mw_mixture(Ta0, Sa0, 
                                                   self.frac_melt)

        # Gety density difference at the initial plume depth
        drho0 = self.get_density_diff(Ta0, self.T0, Sa0, 
                                          self.S0)

        # Calculate initial plume D and U by assuming no initial u
        # upward momentum flux (setting LHS of (2) to 0):

        if self.plume_type == 'line':        
            self.U0 = (drho0*self.g*self.volfl0*self.sinth
                           /(self.e0*self.sinth+self.k))**(1/3) 
            self.D0 = self.volfl0/self.U0
            
        if self.plume_type == 'cone':        
            self.U0 = (np.sqrt(np.pi*self.volfl0/8/self.k**2)
                           *(drho0*self.g*self.sinth))**(2/5) 
            self.D0 = np.sqrt(self.volfl0/(np.pi*self.U0))

        # Return initial state variable
        Yinit = [self.D0*self.U0, 
                 self.D0*self.U0**2, 
                 self.D0*self.U0*self.T0, 
                 self.D0*self.U0*self.S0]

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

#        U_ = max([Y[1] / Y[0], self.u_eps])
  
        U_ = Y[1] / Y[0]


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

    def get_density_diff(self, Ta, T, Sa, S):
        '''
        Get the scaled density difference between the plume (T, S) and 
        the ambient water (Ta, Sa).

        dRho = (rho_plume - rho_amb)/rho_reference
        '''

        dRho = self.BS*(Sa-S)-self.BT*(Ta-T)
        return dRho

###############################################################################


    def get_mw_mixture(self, Ta, Sa, frac_melt):
        '''
        Cooling and freshening ambient water (Ta, Sa) along "Gade lines" (T-S 
        lines resulting from ocean-driven ice melt).
        '''

        # Get local freezing temperature at ambient salinity
        Tf = Sa*self.FPa + self.FPb + self.FPc*self.dep0

        # Get temperature of "effective" end member water mass 
        T_eff = Tf - self.ci/self.c*(Tf - self.Ti)- self.L/self.c

        # Get salinity and temperature at the intersection point
        # between the freezing and Gade lines 
        # ("melt-saturated ambient water")

        S_sat = Sa*(T_eff  - (self.FPb + self.FPc*self.dep0))/(
                                            Sa*self.FPa - Ta + T_eff)
        T_sat = S_sat*self.FPa + self.FPb + self.FPc*self.dep0

        # Compute T, S a fractional distance *frac_melt* along the
        # line between ambient and melt saturated ambient water
        # (along the Gade line towards its intersection with the 
        # freezing line)
    
        Sp = Sa*(1-frac_melt) + S_sat*frac_melt
        Tp = Ta*(1-frac_melt) + T_sat*frac_melt

        return Tp, Sp


###############################################################################
##  EVENT FUNCTIONS (telling solver when to register neutral/terminal depth) ##
###############################################################################

    def event_neutral(self, x, Y):
        '''
        Finds the neutral depth by finding where the density 
        difference is minimized.
        '''
        Ta = self.T_ambient_ip(self.dep0-x)
        Sa = self.S_ambient_ip(self.dep0-x)
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