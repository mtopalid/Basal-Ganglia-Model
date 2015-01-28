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
tr_stop  = 2.5*second
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
gpi      = True
familiar = True
learning  = True

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


@clock.at(tr_stop + start)
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
size = int(duration/dt)

decision_time = 0
decision_time_cog = 0
diff_choice = 0
cogchoice = 0
cues_value = np.ones(4) * 0.5
cues_reward = np.array([0.75,0.25,0.0,0.0])
total_good = 0
mot_before_cog = 0

@after(clock.tick)
def register(t):
    global ind, decision_time, c1, c2, m1, m2, total_good, cogchoice, decision_time_cog, mot_before_cog, diff_choice

    ind = int(t*1000)
    timesteps[ind] = t
    motor[0,:,ind] = Cortex_mot['U'].ravel()
    cognitive[0,:,ind] = Cortex_cog['U'].ravel()

    mot_choice = np.argmax(Cortex_mot['U'])
    cog_choice = np.argmax(Cortex_cog['U'])

    # Check if Cognitive took a decision
    if abs(Cortex_cog['U'].max() - Cortex_cog['U'].min()) > 40.0 and cogchoice < 1:
    	decision_time_cog = t - start
    	cogchoice = 1

    # Check if Motor took a decision
    if abs(Cortex_mot['U'].max() - Cortex_mot['U'].min()) > 40.0:
		cogchoice = 0
		decision_time = t - start

		if mot_choice == m1:
			cgchoice = c1
			mchoice = m1
		elif mot_choice == m2:
			cgchoice = c2
			mchoice = m2

		if not learning:
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



		if learning:

        	#Striatal Learning
			# Compute reward
			reward = np.random.uniform(0,1) < cues_reward[cgchoice]

			# Compute prediction error
			error = reward - cues_value[cgchoice]

			# Update cues values
			cues_value[cgchoice] += error* alpha_c

			# Learn
			lrate = alpha_LTP if error > 0 else alpha_LTD

			dw = error * lrate * Striatum_cog['U'][cgchoice][0]

			w = clip(W_str.weights[cgchoice, cgchoice] + dw, Wmin, Wmax)
			W_str.weights[cgchoice,cgchoice] = w


			if reward:
				# Cortical Learning

				max_cx_ass = np.max(Cortex_ass['U'][cgchoice]) # in order to have no differences between the positions
				y = np.dot(W_cx.weights[cgchoice][cgchoice*4:(cgchoice + 1) * 4].reshape((1,4)), np.ones((4,1))*max_cx_ass)
				dw_Cortex = 10**(-5) * y * Cortex_ass['U'][cgchoice]
				w = clip(W_cx.weights[cgchoice][cgchoice*4:(cgchoice + 1) * 4] + dw_Cortex, 0.095, 0.105)
				W_cx.weights[cgchoice][cgchoice*4:(cgchoice + 1) * 4] = w

		end()


# Run simulation
learning_trials = 40
testing_trials = 20
simulations = 20
np.random.seed(123)
pth = 'Results-Start' + str(start) + '-' + str(learning_trials) + '_' + str(K_cog_ass)
rt = pth + '/RTmean'
if not os.path.exists(rt):
	os.makedirs(rt)

display = False
printt = False
save = True
RTmean_Fam_GPi = np.zeros((simulations,2))
RTmean_UnFam_GPi = np.zeros((simulations,2))
RTmean_Fam = np.zeros((simulations,2))
RTmean_UnFam = np.zeros((simulations,2))
RTmean_Fam_GPi_cog = np.zeros((simulations,2))
RTmean_UnFam_GPi_cog = np.zeros((simulations,2))
RTmean_Fam_cog = np.zeros((simulations,2))
RTmean_UnFam_cog = np.zeros((simulations,2))
total_good_Fam_GPi = np.zeros((simulations,1))
total_good_UnFam_GPi = np.zeros((simulations,1))
total_good_Fam = np.zeros((simulations,1))
total_good_UnFam = np.zeros((simulations,1))

for simulation in range(simulations):

	print "Simulation ", simulation + 1

	gpi      	= True
	familiar 	= True
	learning  	= True
	change 		= True

	size = int(duration/dt) + 1
	timesteps   = np.zeros(size)
	motor       = np.zeros((5, n, size))
	cognitive   = np.zeros((5, n, size))


	init_weights(W_str)
	D = np.zeros(learning_trials)
	Dcog = np.zeros(learning_trials)
	W_cx.weights[np.where(W_cx.weights !=0)] = 0.1
	init_weights(W_cx, Wmin = 0.95, Wmax = 1.05)
	W_str.weights[np.where(W_str.weights !=0)] = 1.0
	init_weights(W_str)

	path = pth + '/simulation_' + str(simulation+1)
	if not os.path.exists(path):
		os.makedirs(path)


	for i in range(learning_trials + testing_trials*4):

		if change:
			reset(GPic.weights, GPim.weights, change = True)
			change = False
		else:
			reset()

		run(time=duration, dt=dt)

		# Training session
		# Learns the two cues
		if i < learning_trials:
			if i == 0:
				print "Learning\n--------"
			D[i] = decision_time
			Dcog[i] = decision_time_cog

			if i == learning_trials - 1:

				print "Cortex"
				print "[%.4f %.4f %.4f %.4f" %(W_cx.weights[0][0], W_cx.weights[0][1], W_cx.weights[0][2], W_cx.weights[0][3])
				print " %.4f %.4f %.4f %.4f" %(W_cx.weights[1][4], W_cx.weights[1][5], W_cx.weights[1][6], W_cx.weights[1][7])
				print " %.4f %.4f %.4f %.4f" % (W_cx.weights[2][8], W_cx.weights[2][9], W_cx.weights[2][10], W_cx.weights[2][11])
				print " %.4f %.4f %.4f %.4f]" %(W_cx.weights[3][12], W_cx.weights[3][13], W_cx.weights[3][14], W_cx.weights[3][15])
				print "Striatum"
				print "[%.4f %.4f %.4f %.4f]\n" % (W_str.weights[0][0], W_str.weights[1][1], W_str.weights[2][2], W_str.weights[3][3])
				learning  = False
				D = np.zeros(testing_trials)
				Dcog = np.zeros(testing_trials)
				total_good = 0
				mot_before_cog = 0
				diff_choice = 0

		# Test of the known cues with gpi
		elif i > learning_trials-1 and i < learning_trials + testing_trials:
			if i == learning_trials:
				print
			D[i-learning_trials] = decision_time
			Dcog[i-learning_trials] = decision_time_cog
			if i == learning_trials + testing_trials - 1:

				d = D[np.nonzero(D)]
				print "Familiar GPi mot: \t\t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_Fam_GPi[simulation,:] = np.array([d.mean(), d.std()])
				d = Dcog[np.nonzero(Dcog)]
				print "Familiar GPi cog: \t\t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_Fam_GPi_cog[simulation,:] = np.array([d.mean(), d.std()])


				familiar = False
				D = np.zeros(testing_trials)
				Dcog = np.zeros(testing_trials)
				total_good_Fam_GPi[simulation] = total_good*100./testing_trials
				print "Performance: \t\t\t", total_good*100./testing_trials, "%"
				print "Motor Decision before Cogn: \t", mot_before_cog
				print "Different Choices: \t\t", diff_choice, "\n"
				total_good = 0
				mot_before_cog = 0
				diff_choice = 0

		# Test of the unknown cues with gpi
		elif i > learning_trials + testing_trials - 1 and i < learning_trials + 2*testing_trials:

			D[i-learning_trials-testing_trials] = decision_time
			Dcog[i-learning_trials-testing_trials] = decision_time_cog
			if i == learning_trials + 2*testing_trials - 1:
				d = D[np.nonzero(D)]
				print "UnFamiliar GPi mot: \t\t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_UnFam_GPi[simulation,:] = np.array([d.mean(), d.std()])
				d = Dcog[np.nonzero(Dcog)]
				print "UnFamiliar GPi cog: \t\t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_UnFam_GPi_cog[simulation,:] = np.array([d.mean(), d.std()])
				total_good_UnFam_GPi[simulation] = total_good*100./testing_trials
				print "Performance: \t\t\t", total_good*100./testing_trials, "%"
				print "Motor Decision before Cogn: \t", mot_before_cog
				print "Different Choices: \t\t", diff_choice, "\n"
				gpi      = False
				familiar = True
				change	 = True
				D = np.zeros(testing_trials)
				Dcog = np.zeros(testing_trials)
				total_good = 0
				mot_before_cog = 0
				diff_choice = 0

		# Test of the known cues without gpi
		elif i > learning_trials + 2*testing_trials - 1 and i < learning_trials + 3*testing_trials:

			D[i-learning_trials-2*testing_trials] = decision_time
			Dcog[i-learning_trials-2*testing_trials] = decision_time_cog
			if i == learning_trials + 3*testing_trials - 1:

				d = D[np.nonzero(D)]
				print "Familiar NoGPi: \t\t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_Fam[simulation,:] = np.array([d.mean(), d.std()])
				d = Dcog[np.nonzero(Dcog)]
				print "Familiar NoGPi cog: \t\t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_Fam_cog[simulation,:] = np.array([d.mean(), d.std()])
				total_good_Fam[simulation] = total_good*100./testing_trials
				print "Performance: \t\t\t", total_good*100./testing_trials, "%"
				print "Motor Decision before Cogn: \t", mot_before_cog
				print "Different Choices: \t\t", diff_choice, "\n"

				familiar = False
				D = np.zeros(testing_trials)
				Dcog = np.zeros(testing_trials)
				total_good = 0
				mot_before_cog = 0
				diff_choice = 0

		# Test of the unknown cues without gpi
		elif i > learning_trials + 3*testing_trials - 1:
			D[i-learning_trials-3*testing_trials] = decision_time
			Dcog[i-learning_trials-3*testing_trials] = decision_time_cog
			if i == learning_trials + 4*testing_trials - 1:
				d = D[np.nonzero(D)]
				print "UnFamiliar NoGPi: \t\t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_UnFam[simulation,:] = np.array([d.mean(), d.std()])
				d = Dcog[np.nonzero(Dcog)]
				print "UnFamiliar NoGPi cog: \t\t%.3f +/- %.3f" % (d.mean(), d.std())
				RTmean_UnFam_cog[simulation,:] = np.array([d.mean(), d.std()])
				total_good_UnFam[simulation] = total_good*100./testing_trials
				print "Performance: \t\t\t", total_good*100./testing_trials, "%"
				print "Motor Decision before Cogn: \t", mot_before_cog
				print "Different Choices: \t\t", diff_choice, "\n"
if save:
	file = rt + '/Fam_GPi_cog.npy'
	np.save(file,RTmean_Fam_GPi_cog)
	file = rt + '/UnFam_GPi_cog.npy'
	np.save(file,RTmean_UnFam_GPi_cog)
	file = rt + '/Fam_cog.npy'
	np.save(file,RTmean_Fam_cog)
	file = rt + '/UnFam_cog.npy'
	np.save(file,RTmean_UnFam_cog)

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
