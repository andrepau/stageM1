import numpy as np
import matplotlib.pyplot as plt
 
def ricker(tp,ts=0.25, a=1, min=0.001, max=2, dt=0.001, tsimu=1.6,grafmax=100):
    """Trace la fonction de Ricker en temps et en frequences. 
    Correspond à la fonction n°6 dans EFISPEC3D (http://efispec.free.fr/docs/html/classmod__source__function.html#a848915deaa3b241fe68fc8671e414962)
    
    ricker(tp,ts,a,min,max,dt,tsimu,grafmax) 
    
    tp = 1/frequence fondamentale
    ts= décalage en temps (time shift) par défaut =0.25s
    a= amplitude facteur, par défaut = 1
    min = temps min, par défaut = 0.001s
    max = temps max, par defaut = 2s
    dt= pas de temps, par defaut = 0.001 s
    tsimu= temps de simulation, par defaut = 1.6 s 
    grafmax=fréquence max sur le graphique, par défaut=100Hz
    """
    
    t = np.arange(min, max, dt)
    y = 2.0*a*((np.pi**2)*((1/(tp**2))*((t-ts)**2))-0.5)* np.exp(-(np.pi**2)*(1/(tp**2))*((t-ts)**2))
    f=1/t
    f1=1/tp
    A=((f/f1)**2)*(1/(np.exp((f/f1)**2)))
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(t,y)
    
    xt=np.linspace(tsimu,tsimu,50)
    yt=np.linspace(-1.5,1.5,50)
    plt.plot(xt,yt)
    
    plt.title('Ricker domaine temporel')
    plt.xlabel('temps (s)')
    plt.ylim(-1.5,1.5)
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(f,A)
    plt.xlim(0,grafmax)
    plt.title('Ricker domaine frequentiel')
    plt.xlabel('frequence (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    print('f1=', f1)
    #return t, y, f, f1, A
    

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def plot_cube(cube_definition):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.1))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)
    
def plot_dispcurve(file, name='file.png'):
    """plot_dispcurve('file.txt', name='file.png')
    
    """
    data = np.loadtxt(file,skiprows = 1)
    
    x = np.reshape(data[:,0],(-1,len(np.unique(data[:,0]))))
    y = 1/(np.reshape(data[:,1],(-1,len(np.unique(data[:,0])))))
    z = np.reshape(data[:,2],(-1,len(np.unique(data[:,0]))))
    
    plt.figure(figsize= (12,7))
    cf = plt.contourf(x,y,z,20, cmap='winter')
    plt.colorbar(label = 'val')
    plt.contour(cf,colors = 'k',linewidths = 0.5,alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(name)
    
    
def plot_dispcurve_lim(file, name='file.png',plotvmin=100,plotvmax=1000,plotfmin=10 ,plotfmax=150, L=24, x1=6, dx=1):
    """plot_dispcurve_lim('file.txt', name='file.png')
    
    plotvmin= vitesse min sur le graph (pour les lim), par défault 10
    plotvmax= vitesse max sur le graph (pour les lim), par défault 1000
    plotfmin= fréquence min sur le graph(pour les lim), par défault 10
    plotfmax= fréquence max sur le graph (pour les lim), par défault 150
    
    L= longueur du dispo, par défaut 24m
    x1= distance source premier recepteur, par defaut 6m 
    dx= distance entre recepteurs, par defaut 1m
    
    """
    data = np.loadtxt(file,skiprows = 1)
    
    x = np.reshape(data[:,0],(-1,len(np.unique(data[:,0]))))
    y = 1/(np.reshape(data[:,1],(-1,len(np.unique(data[:,0])))))
    z = np.reshape(data[:,2],(-1,len(np.unique(data[:,0]))))
    
    plt.figure(figsize= (12,7))
    plt.grid()
    cf = plt.contourf(x,y,z,20, cmap='winter')
    plt.colorbar(label = 'val')
    plt.contour(cf,colors = 'k',linewidths = 0.5,alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    
    #courbes limites
    wlmax1=L  #wlmax est la longueur d'onde max que l'on peux mesurer de façon fiable, au delà non fiable
    print('wlmax1 (pour wl max < L) =', wlmax1, 'm')

    wlmax2=2*x1
    print('wlmax2 (pour wl max < 2*x1) =', wlmax2, 'm')

    wlmax3=0.4*L
    print('wlmax3 (wl max < 0.4L) =', wlmax3, 'm')
    
    wlmax4=0.5*L
    print('wlmax4 (edf : wl< 0.5L)=', wlmax4, 'm')

    wlmin=2*dx
    print('wlmin (pour aliasing quand wl < 2*dx)=', wlmin, "m")
    
    wlmin2=dx
    print('wlmin2 (edf wl < dx)=', wlmin2, "m")

    #plot
    v=np.arange(plotvmin,plotvmax,10)         
    fmin1=v/wlmax1
    fmin2=v/wlmax2
    fmin3=v/wlmax3
    fmin4=v/wlmax4
    fmax=v/wlmin
    fmax2=v/wlmin2

    plt.plot(fmin1,v, label="wl max < L") #ne pas oublier de changer la legende
    plt.plot(fmin2,v,  label="wl max < 2*x1")
    plt.plot(fmin3,v,  label="wl max < 0.4L ")
    plt.plot(fmin4,v, '--', label="edf : wl< 0.5L")
    plt.plot(fmax,v, label="Aliasing quand wl < 2*dx")
    plt.plot(fmax2,v, label="EDF wl< dx")
    plt.xlim(plotfmin,plotfmax)
    plt.ylim(plotvmin,plotvmax)
    plt.xlabel('frequences (Hz)')
    plt.ylabel('vitesse (m/s)')
    plt.legend(bbox_to_anchor=(1.5, 1), loc=2, borderaxespad=0.)
    
    plt.savefig(name)
    
def geom(x,y,z,h1,recepx,xneg,recepy):
    plt.figure(figsize=(15,15))
    plt.subplot(221)
    plt.plot(recepx+xneg,recepy+(y/2),'.')
    sourcex=xneg
    sourcey=y/2
    plt.plot(sourcex,sourcey,'.')

    plt.title("Zoom dispositif d'acquisition")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid()

    plt.subplot(222)
    plt.plot(recepx+xneg,recepy+(y/2),'.')
    sourcex=xneg
    sourcey=y/2
    plt.plot(sourcex,sourcey,'.')

    plt.title("Vu du dessus")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid()

    xcube = [x, 0, 0, x, x]
    ycube = [y, y, 0, 0, y]
    plt.plot(xcube, ycube)

    plt.subplot(224)
    xcube = [x, 0, 0, x, x]
    ycube = [-z, -z, 0, 0, -z]
    plt.plot(xcube, ycube)


    xcubeh = [x, 0, 0, x, x]
    ycubeh = [-h1, -h1, 0, 0, -h1]
    plt.plot(xcubeh, ycubeh)

    plt.title("Vu de côté")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.axis('equal')

    plt.grid()
    
from numpy.fft import fft 
from numpy.fft import ifft     

def plot_fft(signal,t):
    plt.figure(figsize=(20,10))
    plt.subplot(211)
    plt.plot(t,signal)
    plt.grid()
    plt.xlabel('temps (s)')
    plt.ylabel('Amplitude')
    
    tfd = fft(signal)
    N=len(signal)
    spectre = np.absolute(tfd)*2/N
    T=max(t)

    freq=np.arange(N)*1.0/T
    

    plt.subplot(212)
    plt.plot(freq,spectre,'r')
    plt.xlabel('fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(-5,100)
    plt.grid()
    
def plot_fft_reverse(A,freq):
    plt.figure(figsize=(20,10))
    
    N=len(A)
    spectre = np.absolute(A)*2/N
    
    plt.subplot(211)
    plt.plot(freq,spectre,'r')
    plt.grid()
    plt.xlabel('fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(-5,100)
    
    signal = ifft(A)
    
    spectre = np.absolute(signal)*2/N
    T=max(freq)

    t=np.arange(N)*1.0/T
    

    plt.subplot(212)
    plt.plot(t,signal)
    plt.xlabel('temps (s)')
    plt.ylabel('Amplitude')
    
    plt.grid()