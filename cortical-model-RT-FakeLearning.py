# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors:
#  * Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#  * Meropi Topalidou (Meropi.Topalidou@inria.fr)
# -----------------------------------------------------------------------------

# Packages import
import sys
from dana import *
import matplotlib.pyplot as plt
import os

# Simulation parameters
# -----------------------------------------------------------------------------
# Population size
n = 4

# Trial duration
duration = 3.0*second
tr_stop = 2.5*second
# Default Time resolution
dt = 1.0*millisecond

# Initialization of the random generator
np.random.seed(123)


# Threshold
# -------------------------------------
Cortex_h   =  -3.0
Striatum_h =   0.0
STN_h      = -10.0
GPi_h      =  10.0
Thalamus_h = -40.0

# Time constants
# -------------------------------------
tau = 0.01
Cortex_tau   = tau #0.01#tau #
Striatum_tau = tau #0.01
STN_tau      = tau #0.01
GPi_tau      = tau #0.01
Thalamus_tau = tau #0.01
# Noise leve (%)
# -------------------------------------
Cortex_N   =   0.01
Striatum_N =   0.001
STN_N      =   0.001
GPi_N      =   0.03
Thalamus_N =   0.001

# Sigmoid parameters
# -------------------------------------
Vmin       =   0.0
Vmax       =  20.0
Vh         =  16.0
Vc         =   3.0

# Learning parameters
decision_threshold = 40
alpha_c     = 0.05
alpha_LTP  = 0.002
alpha_LTD  = 0.001
Wmin, Wmax = 0.25, 0.75

# Setup
# -------------------------------------
gpi      = True
familiar = True

# Paper
# With GPi (RT / 20 trials)
# familiar stimuli:   418.8 +/- 4 ms
# unfamiliar stimuli: 447.6 +/- 5.6 ms

# Without GPi (RT / 20 trials)
# familiar stimuli:   452.5 +/- 4.2 ms
# unfamiliar stimuli: 495.7 +/- 6.5 ms


# With GPi (RT / 50 trials)
# familiar stimuli:   0.296 +/- 0.035
# unfamiliar stimuli: 0.477 +/- 0.138

# Without GPi (RT / 50 trials)
# familiar stimuli:   0.385 +/- 0.055
# unfamiliar stimuli: 0.718 +/- 0.222

# With GPi (RT / 120 trials)
# HC: 0.308 +/- 0.068
# NC: 0.486 +/- 0.132

# Without GPi (RT / 120 trials)
# HC: 0.399 +/- 0.055
# NC: 0.678 +/- 0.168



# Helper functions
# -----------------------------------------------------------------------------
def sigmoid(V,Vmin=Vmin,Vmax=Vmax,Vh=Vh,Vc=Vc):
    return  Vmin + (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))

def noise(Z, level):
    Z = (1+np.random.uniform(-level/2,level/2,Z.shape))*Z
    return np.maximum(Z,0.0)

def init_weights(L, gain=1, Wmin = Wmin, Wmax = Wmax):

    W = L._weights
    N = np.random.normal(0.5, 0.005, W.shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    L._weights = gain*W*(Wmin + (Wmax - Wmin)*N)

def reset(GPic = [], GPim = [], change = False):
    clock.reset()
    for group in network.__default_network__._groups:
        group['U'] = 0
        group['V'] = 0
        group['I'] = 0
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0
    if change:
		if gpi:
			for j in range(n):
				GPic[j,j] = -0.5
				GPim[j,j] = -0.5
		else:
			GPic[:] = 0
			GPim[:] = 0


def clip(V, Vmin, Vmax):
    return np.minimum(np.maximum(V, Vmin), Vmax)

def figure(timesteps, cognitive, motor, start):

    fig = plt.figure(figsize=(12,5))
    plt.subplots_adjust(bottom=0.15)

    fig.patch.set_facecolor('.9')
    #ax = plt.subplot(2,2,number)

    plt.plot(timesteps[:-1], cognitive[0,0][:-1],c='b', label="Cognitive Cortex")
    plt.plot(timesteps[:-1], cognitive[0,1][:-1],c='b')
    plt.plot(timesteps[:-1], cognitive[0,2][:-1],c='b')
    plt.plot(timesteps[:-1], cognitive[0,3][:-1],c='b')

    plt.plot(timesteps[:-1], motor[0,0][:-1], c='r', label="Motor Cortex")
    plt.plot(timesteps[:-1], motor[0,1][:-1], c='r')
    plt.plot(timesteps[:-1], motor[0,2][:-1], c='r')
    plt.plot(timesteps[:-1], motor[0,3][:-1], c='r')

    plt.xlabel("Time (seconds)")
    plt.ylabel("Activity (Hz)")
    plt.legend(frameon=False, loc='upper left')
    #plt.xlim(0.0,duration)
    plt.ylim(-5.0,60.0)

    if familiar:
       title = " in Habitual Condition"
    else:
        title = " in Novelty Condition"

    if gpi:
        title = "Single trial with GPI" + title
        plt.title(title)
    else:
        title = "Single trial without GPI" + title
        plt.title(title)
    plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', duration])
    # plt.savefig("model-results.png")
    #plt.show()

def subplot(rows,cols,n, alpha=0.0):
    ax = plt.subplot(rows,cols,n)

    #ax.patch.set_alpha(alpha)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="outward")
    return ax

def displayALL(timesteps, cognitive, motor, associative, start):
	fig = plt.figure(figsize=(17, 9))
	plt.subplots_adjust(bottom=0.15, hspace = 0.5)

	associative = associative.reshape(2,n,n,size)


	# STN motor
	# ---------
	ax = subplot(5,3,1)
	ax.set_title("MOTOR", fontsize=24)
	ax.set_ylabel("STN", fontsize=24)
	plt.plot(timesteps[:-1], motor[2,0][:-1],c='r', label="1st")
	plt.plot(timesteps[:-1], motor[2,1][:-1],c='g', label="2nd")
	plt.plot(timesteps[:-1], motor[2,2][:-1],c='b', label="3rd")
	plt.plot(timesteps[:-1], motor[2,3][:-1],c='m', label="4th")


	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', duration])

	# STN cognitive
	# -------------
	ax = subplot(5,3,2)
	ax.set_title("COGNITIVE", fontsize=24)
	plt.plot(timesteps[:-1], cognitive[2,0][:-1],c='r', label="1st")
	plt.plot(timesteps[:-1], cognitive[2,1][:-1],c='g', label="2nd")
	plt.plot(timesteps[:-1], cognitive[2,2][:-1],c='b', label="3rd")
	plt.plot(timesteps[:-1], cognitive[2,3][:-1],c='m', label="4th")


	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])

	ax = subplot(5,3,3,alpha=0)
	ax.set_title("ASSOCIATIVE", fontsize=24)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.spines['left'].set_color('none')
	ax.patch.set_alpha(0.0)

	# Cortex motor
	# ------------
	ax = subplot(5,3,4)
	ax.set_ylabel("CORTEX", fontsize=24)
	plt.plot(timesteps[:-1], motor[0,0][:-1],c='r', label="1st")
	plt.plot(timesteps[:-1], motor[0,1][:-1],c='g', label="2nd")
	plt.plot(timesteps[:-1], motor[0,2][:-1],c='b', label="3rd")
	plt.plot(timesteps[:-1], motor[0,3][:-1],c='m', label="4th")

	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])


	# Cortex cognitive
	# ----------------
	ax = subplot(5,3,5)
	plt.plot(timesteps[:-1], cognitive[0,0][:-1],c='r', label="1st")
	plt.plot(timesteps[:-1], cognitive[0,1][:-1],c='g', label="2nd")
	plt.plot(timesteps[:-1], cognitive[0,2][:-1],c='b', label="3rd")
	plt.plot(timesteps[:-1], cognitive[0,3][:-1],c='m', label="4th")

	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])



	# Cortex associative
	# ------------------
	ax = subplot(5,3,6)
	plt.plot(timesteps[:-1], associative[0,0,0][:-1],c='r', label="0 & 0")
	plt.plot(timesteps[:-1], associative[0,0,1][:-1],c='g', label="0 & 1")
	plt.plot(timesteps[:-1], associative[0,1,0][:-1],c='b', label="0 & 2")
	plt.plot(timesteps[:-1], associative[0,1,1][:-1],c='m', label="0 & 3")
	plt.plot(timesteps[:-1], associative[0,0,2][:-1],c='y', label="1 & 0")
	plt.plot(timesteps[:-1], associative[0,0,3][:-1],c='c', label="1 & 1")
	plt.plot(timesteps[:-1], associative[0,1,2][:-1],c='k', label="1 & 2")
	plt.plot(timesteps[:-1], associative[0,3,3][:-1],c='k', label="3 & 3")

	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])


	# Striatum motor
	# --------------
	ax = subplot(5,3,7)
	ax.set_ylabel("STRIATUM", fontsize=24)
	plt.plot(timesteps[:-1], motor[1,0][:-1],c='r', label="1st")
	plt.plot(timesteps[:-1], motor[1,1][:-1],c='g', label="2nd")
	plt.plot(timesteps[:-1], motor[1,2][:-1],c='b', label="3rd")
	plt.plot(timesteps[:-1], motor[1,3][:-1],c='m', label="4th")

	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])


	# Striatum cognitive
	# ------------------
	ax = subplot(5,3,8)
	plt.plot(timesteps[:-1], cognitive[1,0][:-1],c='r', label="1st")
	plt.plot(timesteps[:-1], cognitive[1,1][:-1],c='g', label="2nd")
	plt.plot(timesteps[:-1], cognitive[1,2][:-1],c='b', label="3rd")
	plt.plot(timesteps[:-1], cognitive[1,3][:-1],c='m', label="4th")

	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])




	# Striatum associative
	# --------------------
	ax = subplot(5,3,9)

	plt.plot(timesteps[:-1], associative[1,0,0][:-1],c='r', label="0 & 0")
	plt.plot(timesteps[:-1], associative[1,0,1][:-1],c='b', label="0 & 1")
	plt.plot(timesteps[:-1], associative[1,1,0][:-1],c='m', label="0 & 2")
	plt.plot(timesteps[:-1], associative[1,1,1][:-1],c='g', label="0 & 3")
	plt.plot(timesteps[:-1], associative[1,0,2][:-1],c='y', label="1 & 0")
	plt.plot(timesteps[:-1], associative[1,0,3][:-1],c='c', label="1 & 1")
	plt.plot(timesteps[:-1], associative[1,1,2][:-1],c='k', label="1 & 2")
	plt.plot(timesteps[:-1], associative[1,3,3][:-1],c='k', label="3 & 3")


	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])

	# GPi motor
	# ---------
	ax = subplot(5,3,10)
	ax.set_ylabel("GPi", fontsize=24)

	plt.plot(timesteps[:-1], motor[3,0][:-1],c='r', label="1st")
	plt.plot(timesteps[:-1], motor[3,1][:-1],c='g', label="2nd")
	plt.plot(timesteps[:-1], motor[3,2][:-1],c='b', label="3rd")
	plt.plot(timesteps[:-1], motor[3,3][:-1],c='m', label="4th")

	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])


	# GPi cognitive
	# -------------
	ax = subplot(5,3,11)
	plt.plot(timesteps[:-1], cognitive[3,0][:-1],c='r', label="1st")
	plt.plot(timesteps[:-1], cognitive[3,1][:-1],c='g', label="2nd")
	plt.plot(timesteps[:-1], cognitive[3,2][:-1],c='b', label="3rd")
	plt.plot(timesteps[:-1], cognitive[3,3][:-1],c='m', label="4th")

	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])

	# Thalamus motor
	# --------------
	ax = subplot(5,3,13)
	ax.set_ylabel("THALAMUS", fontsize=24)
	plt.plot(timesteps[:-1], motor[4,0][:-1],c='r', label="1st")
	plt.plot(timesteps[:-1], motor[4,1][:-1],c='g', label="2nd")
	plt.plot(timesteps[:-1], motor[4,2][:-1],c='b', label="3rd")
	plt.plot(timesteps[:-1], motor[4,3][:-1],c='m', label="4th")

	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])

	# Thalamus cognitive
	# ------------------
	ax = subplot(5,3,14)
	plt.plot(timesteps[:-1], cognitive[4,0][:-1],c='r', label="1st")
	plt.plot(timesteps[:-1], cognitive[4,1][:-1],c='g', label="2nd")
	plt.plot(timesteps[:-1], cognitive[4,2][:-1],c='b', label="3rd")
	plt.plot(timesteps[:-1], cognitive[4,3][:-1],c='m', label="4th")

	plt.xticks([0.0, start, 1.0, 1.5, tr_stop, duration],
               ['0.0',str(start)+'\n(Trial start)','1.0','1.5', str(tr_stop) + '\n(Trial stop)', str(duration)])

	# plt.savefig("model-results-all.pdf")
	#plt.show()


# Populations
# -----------------------------------------------------------------------------
Cortex_cog   = zeros((n,1), """dV/dt = (-V + I + Iext - Cortex_h)/(Cortex_tau);
                           U = noise(V,Cortex_N); I; Iext""")#min_max(V,-3.,60.)
Cortex_mot   = zeros((1,n), """dV/dt = (-V + I + Iext - Cortex_h)/(Cortex_tau);
                           U = noise(V,Cortex_N); I; Iext""")
Cortex_ass   = zeros((n,n), """dV/dt = (-V + I + Iext - Cortex_h)/(Cortex_tau);
                           U = noise(V,Cortex_N); I; Iext""")
Striatum_cog = zeros((n,1), """dV/dt = (-V + I - Striatum_h)/(Striatum_tau);
                           U = noise(sigmoid(V), Striatum_N); I""")
Striatum_mot = zeros((1,n), """dV/dt = (-V + I - Striatum_h)/(Striatum_tau);
                           U = noise(sigmoid(V), Striatum_N); I""")
Striatum_ass = zeros((n,n), """dV/dt = (-V + I - Striatum_h)/(Striatum_tau);
                           U = noise(sigmoid(V), Striatum_N); I""")
STN_cog      = zeros((n,1), """dV/dt = (-V + I - STN_h)/(STN_tau);
                           U = noise(V,STN_N); I""")
STN_mot      = zeros((1,n), """dV/dt = (-V + I - STN_h)/(STN_tau);
                           U = noise(V,STN_N); I""")
GPi_cog      = zeros((n,1), """dV/dt = (-V + I - GPi_h)/(GPi_tau);
                           U = noise(V,GPi_N); I""")
GPi_mot      = zeros((1,n), """dV/dt = (-V + I - GPi_h)/(GPi_tau);
                           U = noise(V,GPi_N); I""")
Thalamus_cog = zeros((n,1), """dV/dt = (-V + I - Thalamus_h)/(Thalamus_tau);
                           U = noise(V,Thalamus_N); I""")
Thalamus_mot = zeros((1,n), """dV/dt = (-V + I - Thalamus_h)/(Thalamus_tau);
                           U = noise(V, Thalamus_N); I""")


# Connectivity
# -----------------------------------------------------------------------------
W_str = DenseConnection( Cortex_cog('U'),   Striatum_cog('I'), 1.0)
init_weights(W_str)

L = DenseConnection( Cortex_mot('U'),   Striatum_mot('I'), 1.0)
init_weights(L)
L = DenseConnection( Cortex_ass('U'),   Striatum_ass('I'), 1.0)
init_weights(L)
L = DenseConnection( Cortex_cog('U'),   Striatum_ass('I'), np.ones((1,2*n+1)))
init_weights(L,0.2)
L = DenseConnection( Cortex_mot('U'),   Striatum_ass('I'), np.ones((2*n+1,1)))
init_weights(L,0.2)

DenseConnection( Cortex_cog('U'),   STN_cog('I'),       1.0 )
DenseConnection( Cortex_mot('U'),   STN_mot('I'),       1.0 )
DenseConnection( Striatum_cog('U'), GPi_cog('I'),      -2.0 )
DenseConnection( Striatum_mot('U'), GPi_mot('I'),      -2.0 )
DenseConnection( Striatum_ass('U'), GPi_cog('I'),      -2.0*np.ones((1,2*n+1)))
DenseConnection( Striatum_ass('U'), GPi_mot('I'),      -2.0*np.ones((2*n+1,1)))
DenseConnection( STN_cog('U'),      GPi_cog('I'),       1.0*np.ones((2*n+1,1)) )
DenseConnection( STN_mot('U'),      GPi_mot('I'),       1.0*np.ones((1,2*n+1)) )
DenseConnection( Cortex_cog('U'),   Thalamus_cog('I'),  0.1 )
DenseConnection( Cortex_mot('U'),   Thalamus_mot('I'),  0.1)

# Faster RT with GPi
# DenseConnection( Thalamus_cog('U'), Cortex_cog('I'),    0.4)
# DenseConnection( Thalamus_mot('U'), Cortex_mot('I'),    0.4)

# Slower RT with GPi
DenseConnection( Thalamus_cog('U'), Cortex_cog('I'),    0.3)
DenseConnection( Thalamus_mot('U'), Cortex_mot('I'),    0.3)

GPic = DenseConnection( GPi_cog('U'),      Thalamus_cog('I'), -0.5 )
GPim = DenseConnection( GPi_mot('U'),      Thalamus_mot('I'), -0.5 )

K_cog_cog = -0.5 * np.ones((2*n+1,1))
K_cog_cog[n,0] = 0.5
K_mot_mot = -0.5 * np.ones((1,2*n+1))
K_mot_mot[0,n] = 0.5
K_ass_ass = -0.5 * np.ones((2*n+1,2*n+1))
K_ass_ass[0,n] = 0.5
K_cog_ass = 0.20
K_mot_ass = 0.01
K_ass_cog = 0.10 * np.ones((1,2*n + 1))
K_ass_mot = 0.10 * np.ones((2*n + 1, 1))

DenseConnection( Cortex_cog('U'), Cortex_cog('I'), K_cog_cog)
DenseConnection( Cortex_mot('U'), Cortex_mot('I'), K_mot_mot)
DenseConnection( Cortex_ass('U'), Cortex_ass('I'), K_ass_ass)
DenseConnection( Cortex_cog('U'), Cortex_ass('I'), K_cog_ass)
DenseConnection( Cortex_mot('U'), Cortex_ass('I'), K_mot_ass)
DenseConnection( Cortex_ass('U'), Cortex_mot('I'), K_ass_mot)
W_cx = DenseConnection( Cortex_ass('U'), Cortex_cog('I'), K_ass_cog)
init_weights(W_cx, Wmin = 0.95, Wmax = 1.05)


start = 500*millisecond
#duration = 2.3 * second + start
# Trial setup
@clock.at(start)
def set_trial(t):
    global c1, c2, m1, m2

    if familiar:
        c1,c2 = 0,1
    else:
        c1,c2 = 2,3

    if familiar:
    	temp = np.array([0,1])
        np.random.shuffle(temp)
        m1, m2 = 0,1#temp
    else:
    	temp = np.array([2,3])
        np.random.shuffle(temp)
        m1, m2 = 2,3#temp

    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0

    v = 7
    Cortex_mot['Iext'][0,m1]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_mot['Iext'][0,m2]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_cog['Iext'][c1,0]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_cog['Iext'][c2,0]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_ass['Iext'][c1,m1] = v + np.random.normal(0,v*Cortex_N)
    Cortex_ass['Iext'][c2,m2] = v + np.random.normal(0,v*Cortex_N)


@clock.at(2.0*second + start)
def reset_trial(t):
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0

# Learning
# -----------------------------------------------------------------------------

size = int(duration/dt)
timesteps   = np.zeros(size)
motor       = np.zeros((5, n, size))
cognitive   = np.zeros((5, n, size))
associative = np.zeros((2, n*n, size))
@after(clock.tick)
def register(t):
    index = int(t*1000) - 1

    timesteps[index] = t

    motor[0,:,index] = Cortex_mot['U'].ravel()
    motor[1,:,index] = Striatum_mot['U'].ravel()
    motor[2,:,index] = STN_mot['U'].ravel()
    motor[3,:,index] = GPi_mot['U'].ravel()
    motor[4,:,index] = Thalamus_mot['U'].ravel()

    cognitive[0,:,index] = Cortex_cog['U'].ravel()
    cognitive[1,:,index] = Striatum_cog['U'].ravel()
    cognitive[2,:,index] = STN_cog['U'].ravel()
    cognitive[3,:,index] = GPi_cog['U'].ravel()
    cognitive[4,:,index] = Thalamus_cog['U'].ravel()

    associative[0,:,index] = Cortex_ass['U'].ravel()
    associative[1,:,index] = Striatum_ass['U'].ravel()


decision_time = 0
decision_time_cog = 0
diff_choice = 0
cogchoice = 0
motchoice = 0
cues_value = np.ones(4) * 0.5
cues_reward = np.array([0.75,0.25,0.0,0.0])
total_good = 0
mot_before_cog = 0

@after(clock.tick)
def register(t):
    global decision_time, c1, c2, m1, m2, total_good, cogchoice, decision_time_cog, mot_before_cog, diff_choice, motchoice

    mot_choice = np.argmax(Cortex_mot['U'])
    cog_choice = np.argmax(Cortex_cog['U'])

    # Check if Cognitive took a decision
    if abs(Cortex_cog['U'].max() - Cortex_cog['U'].min()) > 40.0 and cogchoice < 1:
    	decision_time_cog = t - start
    	cogchoice = 1

    # Check if Motor took a decision
    if abs(Cortex_mot['U'].max() - Cortex_mot['U'].min()) > 40.0 and motchoice < 1:
		motchoice = 1
		decision_time = t - start

		if mot_choice == m1:
			cgchoice = c1
			mchoice = m1
		elif mot_choice == m2:
			cgchoice = c2
			mchoice = m2

		# Check if Motor decides before Cognitive
		if decision_time_cog > 0.0:
			mot_before_cog = mot_before_cog + (0 if decision_time_cog < decision_time else 1)
		else:
			mot_before_cog = mot_before_cog + 1


		# Check if Motor and Congitive take same decision
		cog_choice = np.argmax(Cortex_cog['U'])
		if not cog_choice == cgchoice:
			diff_choice += 1
			#print "Motor and Cognitive didn't choose the same"

		if cgchoice == min(c1,c2):
			total_good += 1


# Run simulation
testing_trials = 1
W_cx.weights[0] *= 1.03
W_str.weights[0] *= 1.30
print "Cortex"
print "[%.4f %.4f %.4f %.4f" %(W_cx.weights[0][0], W_cx.weights[0][1], W_cx.weights[0][2], W_cx.weights[0][3])
print " %.4f %.4f %.4f %.4f" %(W_cx.weights[1][4], W_cx.weights[1][5], W_cx.weights[1][6], W_cx.weights[1][7])
print " %.4f %.4f %.4f %.4f" % (W_cx.weights[2][8], W_cx.weights[2][9], W_cx.weights[2][10], W_cx.weights[2][11])
print " %.4f %.4f %.4f %.4f]" %(W_cx.weights[3][12], W_cx.weights[3][13], W_cx.weights[3][14], W_cx.weights[3][15])
print "Striatum"
print "[%.4f %.4f %.4f %.4f]\n" % (W_str.weights[0][0], W_str.weights[1][1], W_str.weights[2][2], W_str.weights[3][3])

reset()
run(time=duration, dt=dt)

figure(timesteps, cognitive, motor, start)
displayALL(timesteps, cognitive, motor, associative, start)
print "Familiar GPi mot: \t\t%.3f" % (decision_time)
print "Familiar GPi cog: \t\t%.3f" % (decision_time_cog)
print "Performance: \t\t\t", total_good*100./testing_trials, "%"
print "Motor Decision before Cogn: \t", mot_before_cog
print "Different Choices: \t\t", diff_choice, "\n"
total_good = 0
mot_before_cog = 0
diff_choice = 0
cogchoice = 0
motchoice = 0

familiar = False
reset()
run(time=duration, dt=dt)
figure(timesteps, cognitive, motor, start)
displayALL(timesteps, cognitive, motor, associative, start)
print "UnFamiliar GPi mot: \t\t%.3f" % (decision_time)
print "UnFamiliar GPi cog: \t\t%.3f" % (decision_time_cog)
print "Performance: \t\t\t", total_good*100./testing_trials, "%"
print "Motor Decision before Cogn: \t", mot_before_cog
print "Different Choices: \t\t", diff_choice, "\n"
total_good = 0
mot_before_cog = 0
diff_choice = 0
cogchoice = 0
motchoice = 0

gpi      = False
familiar = True
reset(GPic.weights, GPim.weights, change = True)
run(time=duration, dt=dt)
figure(timesteps, cognitive, motor, start)
print "Familiar NoGPi: \t\t%.3f" % (decision_time)
print "Familiar NoGPi cog: \t\t%.3f" % (decision_time_cog)
print "Performance: \t\t\t", total_good*100./testing_trials, "%"
print "Motor Decision before Cogn: \t", mot_before_cog
print "Different Choices: \t\t", diff_choice, "\n"
total_good = 0
mot_before_cog = 0
diff_choice = 0
cogchoice = 0
motchoice = 0

familiar = False
reset()
run(time=duration, dt=dt)
figure(timesteps, cognitive, motor, start)
print "UnFamiliar NoGPi: \t\t%.3f" % (decision_time)
print "UnFamiliar NoGPi cog: \t\t%.3f" % (decision_time_cog)
print "Performance: \t\t\t", total_good*100./testing_trials, "%"
print "Motor Decision before Cogn: \t", mot_before_cog
print "Different Choices: \t\t", diff_choice, "\n"

plt.show()
