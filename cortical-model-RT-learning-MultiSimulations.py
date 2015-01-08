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

# Default Time resolution
dt = 1.0*millisecond

# Initialization of the random generator
#  -> reproductibility
#np.random.seed(1)


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
display  = True
cortical = True
gpi      = True
familiar = True
learning  = True
fake_learning = False

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

def init_weights(L, gain=1, Wmin = 0.25, Wmax = 0.75):

    W = L._weights
    N = np.random.normal(0.5, 0.005, W.shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    L._weights = gain*W*(Wmin + (Wmax - Wmin)*N)

def reset():
    clock.reset()
    for group in network.__default_network__._groups:
        group['U'] = 0
        group['V'] = 0
        group['I'] = 0
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0

def clip(V, Vmin, Vmax):
    return np.minimum(np.maximum(V, Vmin), Vmax)

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
#init_weights(W_str)

# Simulate basal learning
if fake_learning:
	W_str.weights[0] *= 1.50
	W_str.weights[1] *= 1.25

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

if gpi:
    DenseConnection( GPi_cog('U'),      Thalamus_cog('I'), -0.5 )
    DenseConnection( GPi_mot('U'),      Thalamus_mot('I'), -0.5 )

if cortical:
    K_cog_cog = -0.5 * np.ones((2*n+1,1))
    K_cog_cog[n,0] = 0.5
    K_mot_mot = -0.5 * np.ones((1,2*n+1))
    K_mot_mot[0,n] = 0.5
    K_ass_ass = -0.5 * np.ones((2*n+1,2*n+1))
    K_ass_ass[0,n] = 0.5
    K_cog_ass = 0.20
    K_mot_ass = 0.01
    K_ass_cog = 0.10 * np.ones((1,2*n + 1))
    K_ass_mot = 0.1 * np.ones((2*n + 1, 1))

    DenseConnection( Cortex_cog('U'), Cortex_cog('I'), K_cog_cog)
    DenseConnection( Cortex_mot('U'), Cortex_mot('I'), K_mot_mot)
    DenseConnection( Cortex_ass('U'), Cortex_ass('I'), K_ass_ass)
    DenseConnection( Cortex_cog('U'), Cortex_ass('I'), K_cog_ass)
    DenseConnection( Cortex_mot('U'), Cortex_ass('I'), K_mot_ass)
    DenseConnection( Cortex_ass('U'), Cortex_mot('I'), K_ass_mot)
    W_cx = DenseConnection( Cortex_ass('U'), Cortex_cog('I'), K_ass_cog)
    #init_weights(W_cx, Wmin = 0.95, Wmax = 1.05)


    # Simulate cortical learning

    if fake_learning:
		W_cx.weights[0] *= 1.05
		W_cx.weights[1] *= 1.025


# Trial setup
@clock.at(500*millisecond)
def set_trial(t):
    global c1, c2, m1, m2

    if familiar:
        c1,c2 = 0,1
    else:
        c1,c2 = 2,3

    if familiar:
    	temp = np.array([0,1])
        np.random.shuffle(temp)
        m1, m2 = temp
    else:
    	temp = np.array([2,3])
        np.random.shuffle(temp)
        m1, m2 = temp

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


@clock.at(2500*millisecond)
def reset_trial(t):
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0

# Learning
# -----------------------------------------------------------------------------
P, R = [], []
bad = 0
diff_choice = 0

size = int(duration/dt) + 1

timesteps   = np.zeros(size)
motor       = np.zeros((5, n, size))
cognitive   = np.zeros((5, n, size))
associative = np.zeros((2, n*n, size))

index       = 0
decision_time = 0

cues_value = np.ones(4) * 0.5
cues_reward = np.array([0.75,0.25,0.0,0.0])
total_good = 0
@after(clock.tick)
def register(t):
    global index, decision_time, c1, c2, m1, m2, total_good

    # timesteps[index] = t

    mot_choice = np.argmax(Cortex_mot['U'])
    cog_choice = np.argmax(Cortex_cog['U'])
    if abs(Cortex_mot['U'].max() - Cortex_mot['U'].min()) > 40.0:
        decision_time = t - 500*millisecond
        if mot_choice == m1:
			cgchoice = c1
			mchoice = m1
        elif mot_choice == m2:
			cgchoice = c2
			mchoice = m2
        same = cgchoice == cog_choice
        good = 1 if c1==cgchoice else 0
        total_good += good
        #print "cgchoice = ", cgchoice, " good = ", good, " total_good = ", total_good
        if 0:
			if cog_choice == c1:
				cgchoice = c1
			elif cog_choice == c2:
				cgchoice = c2

        if cgchoice == min(c1,c2):
			P.append(1)
        else:
			P.append(0)

		# Compute reward
        reward = np.random.uniform(0,1) < cues_reward[mchoice]
        R.append(reward)

        if learning:
			# Compute prediction error
			#error = cues_reward[choice] - cues_value[choice]
			error = reward - cues_value[mchoice]

			# Update cues values
			cues_value[mchoice] += error* alpha_c

			# Learn
			lrate = alpha_LTP if error > 0 else alpha_LTD

			dw = error * lrate * Striatum_cog['U'][mchoice][0]

			#if dw < 0.0001:
			#	print "i = ", i

			#W = W_cortex_cog_to_striatum_cog
			w = clip(W_str.weights[mchoice, mchoice] + dw, Wmin, Wmax)
			W_str.weights[mchoice,mchoice] = w


        if reward:

			max_cx_ass = np.max(Cortex_ass['U'][mchoice])
			y = np.dot(W_cx.weights[mchoice][mchoice*4:(mchoice + 1) * 4].reshape((1,4)), np.ones((4,1))*max_cx_ass)
			#y = Cortex_cog['U'][mchoice][0]
			dw_Cortex = 10**(-5) * max_cx_ass * y
			w = clip(W_cx.weights[mchoice][mchoice*4:(mchoice + 1) * 4] + dw_Cortex, 0.095, 0.105)
			W_cx.weights[mchoice][mchoice*4:(mchoice + 1) * 4] = w

        if 0:
			# Just for displaying ordered cue

			if cgchoice == c1:
				print "Cognitive Choice:          [%d] / %d  (good)" % (c1, c2)
			else:
				print "Cognitive Choice:           %d / [%d] (bad)" % (c1, c2)
			if mchoice == m1:
				print "Motor Choice:    [%d] / %d  " % (m1,m2)
			else:
				print "Motor Choice:     %d / [%d] " % (m1,m2)

        end()

    # motor[0,:,index] = Cortex_mot['U'].ravel()
    # motor[1,:,index] = Striatum_mot['U'].ravel()
    # motor[2,:,index] = STN_mot['U'].ravel()
    # motor[3,:,index] = GPi_mot['U'].ravel()
    # motor[4,:,index] = Thalamus_mot['U'].ravel()
    # cognitive[0,:,index] = Cortex_cog['U'].ravel()
    # cognitive[1,:,index] = Striatum_cog['U'].ravel()
    # cognitive[2,:,index] = STN_cog['U'].ravel()
    # cognitive[3,:,index] = GPi_cog['U'].ravel()
    # cognitive[4,:,index] = Thalamus_cog['U'].ravel()
    # associative[0,:,index] = Cortex_ass['U'].ravel()
    # associative[1,:,index] = Striatum_ass['U'].ravel()
    # index = index + 1


# Run simulation
learning_trials = 60
testing_trials = 20
simulations = 20

rt = 'Results/RTmean'
if not os.path.exists(rt):
	os.makedirs(rt)
RTmean_Fam_GPi = np.zeros((simulations,2))
RTmean_UnFam_GPi = np.zeros((simulations,2))
RTmean_Fam = np.zeros((simulations,2))
RTmean_UnFam = np.zeros((simulations,2))
total_good_Fam_GPi = np.zeros((simulations,1))
total_good_UnFam_GPi = np.zeros((simulations,1))
total_good_Fam = np.zeros((simulations,1))
total_good_UnFam = np.zeros((simulations,1))
save = True
for simulation in range(simulations):

	print "Simulation ", simulation + 1

	gpi      = True
	familiar = True
	learning  = True

	D = np.zeros(learning_trials)
	W_cx.weights[np.where(W_cx.weights !=0)] = 0.1
	init_weights(W_cx, Wmin = 0.95, Wmax = 1.05)
	W_str.weights[np.where(W_str.weights !=0)] = 1.0
	init_weights(W_str)

	path = 'Results/simulation_' + str(simulation+1)
	if not os.path.exists(path):
		os.makedirs(path)

	for i in range(learning_trials + testing_trials*4):

		# Training session
		# Learns the two cues
		if i < learning_trials:
			D[i] = decision_time
			#print "Trial %d: %.3f" % (i, decision_time)

		# Test of the known cues with gpi
		elif i > learning_trials-1 and i < learning_trials + testing_trials:
			if i == learning_trials:
				if save:
					file = path + '/_Weights_Cortex.npy'
					np.save(file,W_cx.weights)
					file = path + '/_Weights_Striatum.npy'
					np.save(file,W_str.weights)
				print "Cortex:\n", W_cx.weights
				print "Striatum:\n", W_str.weights
				learning  = False
				D = np.zeros(testing_trials)
				print "total_good = ", total_good*100./learning_trials, "\%"
				total_good = 0
			D[i-learning_trials] = decision_time

		# Test of the unknown cues with gpi
		elif i > learning_trials + testing_trials - 1 and i < learning_trials + 2*testing_trials:
			if i == learning_trials + testing_trials:

				d = D[np.nonzero(D)]
				print "Familiar GPi: \t\t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_Fam_GPi[simulation,:] = np.array([d.mean(), d.std()])

				familiar = False
				D = np.zeros(testing_trials)
				print "total_good = ", total_good*100./testing_trials, "%"
				total_good_Fam_GPi[simulation] = total_good*100./testing_trials
				total_good = 0
			D[i-learning_trials-testing_trials] = decision_time

		# Test of the known cues without gpi
		elif i > learning_trials + 2*testing_trials - 1 and i < learning_trials + 3*testing_trials:
			if i ==  learning_trials + 2*testing_trials:
				d = D[np.nonzero(D)]
				print "UnFamiliar GPi: \t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_UnFam_GPi[simulation,:] = np.array([d.mean(), d.std()])

				gpi      = False
				familiar = True
				D = np.zeros(testing_trials)
				print "total_good = ", total_good*100./testing_trials, "%"
				total_good_UnFam_GPi[simulation] = total_good*100./testing_trials
				total_good = 0
			D[i-learning_trials-2*testing_trials] = decision_time

		# Test of the unknown cues without gpi
		elif i > learning_trials + 3*testing_trials - 1:
			if i == learning_trials + 3*testing_trials:
				d = D[np.nonzero(D)]
				print "Familiar NoGPi: \t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_Fam[simulation,:] = np.array([d.mean(), d.std()])

				familiar = False
				D = np.zeros(testing_trials)
				print "total_good = ", total_good*100./testing_trials, "%"
				total_good_Fam[simulation] = total_good*100./testing_trials
				total_good = 0
			D[i-learning_trials-3*testing_trials] = decision_time

		reset()
		run(time=duration, dt=dt)
	d = D[np.nonzero(D)]
	print "UnFamiliar NoGPi: \t%.3f +/- %.3f\n\n" % (d.mean(), d.std())
	RTmean_UnFam[simulation,:] = np.array([d.mean(), d.std()])
	print "total_good = ", total_good*100./testing_trials, "%"
	total_good_UnFam[simulation] = total_good*100./testing_trials
	total_good = 0
if save:
	file = rt + '/Fam_GPi.npy'
	np.save(file,RTmean_Fam_GPi)
	file = rt + '/UnFam_GPi.npy'
	np.save(file,RTmean_UnFam_GPi)
	file = rt + '/Fam.npy'
	np.save(file,RTmean_Fam)
	file = rt + '/UnFam.npy'
	np.save(file,RTmean_UnFam)

	file = rt + '/total_good_Fam_GPi.npy'
	np.save(file,total_good_Fam_GPi)
	file = rt + '/total_good_UnFam_GPi.npy'
	np.save(file,total_good_UnFam_GPi)
	file = rt + '/total_good_Fam.npy'
	np.save(file,total_good_Fam)
	file = rt + '/total_good_UnFam.npy'
	np.save(file,total_good_UnFam)
