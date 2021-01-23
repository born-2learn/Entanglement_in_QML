import numpy as np
from qiskit import *
from qiskit.circuit import Parameter
from math import radians, pow
from qiskit.quantum_info import Statevector, DensityMatrix, entropy
import matplotlib.pyplot as plt

class GradientDescentOptimizer:
    '''
    This class is used to perform Optimization on the input angle for the parameterized circuit using the Gradient Descent Optimizer.

    '''
    def __init__(self, shots):
        self.shots = shots

    def mse_cost_function(self, prob_avg_01, prob_avg_10):
        '''
        This function calculates the cost for the parametric circuit, using Mean Squared Error method

        Parameters:
        -----------
        prob_avg_01              : Probability of the state |01>
        prob_avg_10              : Probability of the state |10>
        '''
        return pow((prob_avg_01-prob_avg_10), 2)

    def unsymmetrical_cost_function(self, prob_avg_01, prob_avg_10):
        '''
        This function calculates the cost for the parametric circuit, when only the |01> + |10> state is required in the output.

        Parameters:
        -----------
        prob_avg_01              : Probability of the state |01>
        prob_avg_10              : Probability of the state |10>
        '''
        return prob_avg_10 - (prob_avg_01+prob_avg_10)/(2)

    def optimize_circuit_sgd(self, qc, quantum_circuit_parameter, angle_degrees, cost_function='mse', learning_rate = 2):
        '''
        This function is used for optimizing the values of angle_degrees, using Gradient Descent method.

        Parameters:
        -----------
        qc                          : Quantum Circuit object
        quantum_circuit_parameter   : parameter object
        angle_degrees               : Angle(in degrees) by which the parameterised gates will rotate
        cost_function               : The type of cost function to be used[default: mse]
        learning_rate               : The learning rate for optimization[default: 2]
        '''
        i = 0
        max_i = 500
        previous_step_size = 1
        precision = -1


        loss_function = []
        vn_entropy = []
        epochs = []
        epoch_var = 0
        while i<max_i and previous_step_size>precision: #iterating over until the error converges
            epoch_var+=1
            epochs.append(epoch_var)

            theta_radians = radians(angle_degrees) #converting the degrees to radians
            previous_angle = angle_degrees

            bell_state = execute(qc, backend = Aer.get_backend('statevector_simulator'), shots = self.shots, parameter_binds=[{quantum_circuit_parameter: theta_radians}]).result().get_statevector()
            #counts = job.result().get_counts()
            psi = Statevector(bell_state)
            counts = psi.probabilities_dict()
            print(counts)

            
            D = DensityMatrix(bell_state)
            vn_entropy_val = entropy(D, base=2)
            vn_entropy.append(vn_entropy_val)

            
            
            #print(counts)
            try:
                prob_avg_01 = counts['00']
                
            except:
                prob_avg_01 = 0
            try:
                prob_avg_10 = counts['11']
            except:
                prob_avg_10 = 0
            
            if cost_function == 'mse':
                loss_function.append(self.mse_cost_function(prob_avg_01, prob_avg_10))
                angle_degrees = angle_degrees - learning_rate*self.mse_cost_function(prob_avg_01, prob_avg_10)

            if cost_function == 'unsymmetrical':
                angle_degrees = angle_degrees - learning_rate*self.unsymmetrical_cost_function(prob_avg_01, prob_avg_10)

            previous_step_size = abs(angle_degrees - previous_angle)
            i+=1
    
        print(angle_degrees)
        
        return angle_degrees, counts, epochs, vn_entropy, loss_function

    def __str__(self):
        return "Gradient Descent Optimizer"


class BellStateCircuit:
    '''
    This class is used to generate a parameterized circuit that will be optimized to form the Bell State circuit with equal 
    probability of |01> and |10> states.
    '''

    def __init__(self):
        self.qc = None
        self.executable = None
        self.theta = 90
    
    def create_circuit(self, shots, angle_degrees= 90):
        '''
        Creates a parameterized circuit with one tunable parameter with angle_degrees.

        Parameters:
        -----------
        shots           : The total number of shots for which the quantum experiment will run
        angle_degrees   : The angle in degrees, that the parameterized gate will be rotated by
        '''
        self.angle_degrees = angle_degrees
        self.parameter = Parameter('param1')

        self.qc = QuantumCircuit(2, 2)
        state1 = [1,0]
        state2 = [1,0]
        self.qc.initialize(state1, 0) #initializing qubit 0 to state 0
        self.qc.initialize(state2, 1) #initialing qubit 1 to state 1
        self.qc.ry(self.parameter, 0)
        self.qc.barrier()
        self.qc.cx(0, 1)
        self.qc.barrier()
        #self.qc.measure(0, 0)
        #self.qc.measure(1, 1)

    def draw_circuit(self):
        '''
        Displays the circuit in text format.
        '''
        self.qc.draw('text')
    
    def extract_circuit(self):
        '''
        This function returns the Quantum Circuit object, parameter object and final angle(in degrees), 
        that will be passed to the GradientDescent class.
        '''
        return self.qc, self.parameter, self.angle_degrees

    def __str__(self):
        return "Bell State Circuit Generator"
