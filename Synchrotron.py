import numpy as np
import Constants as C
import scipy.special as special
import scipy.integrate as integrate

class Synchrotron:
    
    def __init__(self):
        '''save as array and interpolate F(\gamma) instead of calculating integral function every call'''
        F = lambda x: x*integrate.quad( lambda y: special.kv(5.0/3.0,y), x,np.inf )[0]
        self.xary=10**np.linspace(-2,1.3,100)
        self.Fary=np.zeros_like(self.xary);
        for i in range(len(self.Fary)):
            self.Fary[i] = F(self.xary[i])
    
    @staticmethod
    def derivative(y,x=1):
        '''utility function to numerically calculate derivative of array'''
        eps = 1e-3
        dx = np.diff(x)
        dlogx = np.diff(np.log(x))
        if np.all(dx<dx[0]*(1+eps)) and np.all(dx>dx[0]*(1-eps)):
            '''constant spacing in x'''
            dydx = np.gradient(y,edge_order=2)/np.gradient(x)
        elif np.all(dlogx<dlogx[0]*(1+eps)) and np.all(dlogx>dlogx[0]*(1-eps)):
            '''logarithmicly constant spacing in x'''
            dydx = (y/x)*(np.gradient(np.log(y),edge_order=2)/np.gradient(np.log(x)))
        else:
            dydx = np.gradient(y,edge_order=2)/np.gradient(x)
        return dydx
    
    def F(self, x):
        x = np.asarray(x)
        F = np.interp(x, self.xary,self.Fary, left=0.0,right=0.0)
        F[x<self.xary[0]] = self.Fary[0]*(x[x<self.xary[0]]/self.xary[0])**(1.0/3.0)
        F[x>self.xary[-1]] = self.Fary[-1]*(x[x>self.xary[-1]]/self.xary[-1])**0.5*np.exp(-(x[x>self.xary[-1]]-self.xary[-1]))
        return F
    
    def Pnu(self, gamma,B,nu):
        '''synchrotron spectral power at frequency nu for an electron of Lorentz factor gamma gyrating in magnetic field B'''
        #gamma, B, nu = np.asarray(gamma), np.asarray(B), np.asarray(nu)
        nuc = (C.q*B/(2*np.pi*C.me*C.c))*gamma**2
        Pnu = np.zeros_like(nu)
        #for i in range(len(nu)):
        #    Pnu[i] = 2*np.pi*(C.q**3*B/(3.0**0.5*np.pi*C.me*C.c**2))*self.F(nu[i]/nuc)
        Pnu = 2*np.pi*(C.q**3*B/(3.0**0.5*np.pi*C.me*C.c**2))*self.F(nu/nuc)
        return Pnu
    
    def j_nu(self, gamma,Ng,B,nu):
        '''synchrtron emissivity at frequency nu for a distribution of electrons Ng (=dN/dgamma) with Lorentz factors gamma gyrating in magnetic field B'''
        jnu = np.zeros_like(nu)
        for i in range(len(nu)):
            jnu[i] = (4*np.pi)**(-1)*np.trapz( Ng*self.Pnu(gamma,B,nu[i]), x=gamma )
        return jnu
    
    def alpha_nu(self, gamma,Ng,B,nu):
        '''synchrtron absorption coefficient at frequency nu for a distribution of electrons Ng (=dN/dgamma) with Lorentz factors gamma gyrating in magnetic field B'''
        alphanu = np.zeros_like(nu)
        for i in range(len(nu)):
            alphanu[i] = -(8*np.pi*C.me*nu[i]**2)**(-1)*np.trapz( gamma**2*self.Pnu(gamma,B,nu[i])*self.derivative(Ng/gamma**2,x=gamma), x=gamma )
        return alphanu
    
    def tau_nu(self, gamma,Ng,B,nu,R):
        '''synchrtron optical depth at frequency nu for a distribution of electrons Ng (=dN/dgamma) with Lorentz factors gamma gyrating in magnetic field B'''
        return self.alpha_nu(gamma,Ng,B,nu)*R

    # def f_ssa(self, gamma,Ng,B,nu,R):
    #     return (1.-np.exp(tau_nu(gamma,Ng,B,nu,R)))/self.tau_nu(gamma,Ng,B,nu,R)
    
    def S_nu(self, gamma,Ng,B,nu):
        '''source function'''
        return self.j_nu(gamma,Ng,B,nu)/self.alpha_nu(gamma,Ng,B,nu)
    
    def E(self, gamma,gamma_ary,Ng,B,R):
        nuc = (C.q*B/(2*np.pi*C.me*C.c))*gamma**2
        nu = nuc*10**np.linspace(-2,1,2e2)
        E = (2*C.me**2*C.c**2)**(-1)*np.trapz( self.Pnu(gamma,B,nu)*self.S_nu(gamma_ary,Ng,B,nu)*(1.0-np.exp(-self.tau_nu(gamma_ary,Ng,B,nu,R))/nu**2), x=nu )
        return E
    
    def Lnu(self, gamma,Ng,B,nu,R):
        '''synchrotron luminosity [note - not most efficient calculation since alpha_nu is called twice instead of only once]'''
        return (np.pi*(4*np.pi*R**2))*self.S_nu(gamma,Ng,B,nu)*(1.0-np.exp(-self.tau_nu(gamma,Ng,B,nu,R)))
    
    def Synchrotron_of_gammae(self, gamma,Ng,B,nu,R=np.inf):
        '''ignoring absorption'''
        Lnu = np.zeros_like(nu)
        nuc = (C.q*B/(2*np.pi*C.me*C.c))*gamma**2 # cyclotron frequency of each electron Lorentz factor
        Te = gamma*C.me*C.c**2/(3*C.kb) # effective electron temperature
        Lnu_T = np.zeros_like(nu)
        
        for i in range(len(nu)):
            Pnu = np.zeros_like(gamma)
            Pnu = 2*np.pi*(C.q**3*B/(3.0**0.5*np.pi*C.me*C.c**2))*Synchrotron.F(nu[i]/nuc) # contribution of each electron Lorentz factor to emitted power at frequency nu
            Lnu[i] = np.trapz( Pnu*Ng, x=gamma ) # sum over entire electron distribution                                             #Missing 1/4pi?
            Lnu_T[i] = np.interp(nu[i], nuc,4*np.pi*R**2*(2*C.kb*Te*nuc**2/C.c**2)) # Rayleigh-Jeans "thermal" luminosity    
        
        Lnu[Lnu>Lnu_T] = Lnu_T[Lnu>Lnu_T] # if flux > black-body then optically thick => use black-body (approximate treatment of self-absorption)
        
        return Lnu