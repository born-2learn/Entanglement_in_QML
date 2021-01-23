from libraries.simple_variational_circuit import BellStateCircuit, GradientDescentOptimizer

import matplotlib.pyplot as plt

shots_list = [100] #Number of measurements/shots
results = {} #dictionary to store the final results of the observations

if __name__=='__main__':
    '''
    This is the main function of the entire project that is used to generate an equal probabilty
    of |01> and |10> states, after optimizing the parameters of a parameterized quantum circuit.
    '''
    #print('Enter initial Angle(in degrees): ')
    #angle_degrees = input()
    angle_degrees = 0 #initial angle taken
    

    for shots in shots_list:
        if shots ==1:
            results[shots] = [90.000000000, {'01':1}]
            continue
        elif shots == 10:
            learning_rate = 0.38 #setting learning rate for shots = 10
        elif shots == 100:
            learning_rate = 1.29
        else:
            learning_rate = 3.2
        
        #Creation of the parameterized circuit
        quantum_circuit = BellStateCircuit()
        print(quantum_circuit)
        quantum_circuit.create_circuit(shots=shots, angle_degrees= angle_degrees)
        quantum_circuit.draw_circuit()
        quantum_circuit_object, quantum_circuit_parameter, angle_degrees_ckt = quantum_circuit.extract_circuit()

        #Optimization using Gradient Descent
        optimizer = GradientDescentOptimizer(shots=shots)
        print(optimizer)
        angle_degrees_ckt, counts, epochs, vn_entropy, loss_function = optimizer.optimize_circuit_sgd(quantum_circuit_object, quantum_circuit_parameter, angle_degrees_ckt, learning_rate = learning_rate)
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss Function')
        plt.title('Cost Function')
        plt.plot(epochs, loss_function, color='tab:blue')
        #plot1 = plt.figure(1)
        plt.show()

        plt.xlabel('Epochs')
        plt.ylabel('von Neumann Entropy')
        plt.title('S(X)')
        plt.plot(epochs, vn_entropy, color='tab:blue')
        #plot1 = plt.figure(1)
        plt.show()

        plt.xlabel('Loss Function')
        plt.ylabel('von Neumann Entropy')
        plt.title('vN entropy vs loss')
        plt.plot( loss_function, vn_entropy, color='tab:blue')
        #plot2 = plt.figure(2)
        plt.show()

        results[shots] = [angle_degrees_ckt, counts]

        #Resetting all varuiables for the next iteration
        quantum_circuit = None
        optimizer = None
        quantum_circuit_object, quantum_circuit_parameter, angle_degrees_ckt = None, None, None
        counts = None
        
    for i in results:
        print('Number of measurements:\t',i,'\tFinal parameter(angle in degrees):\t', results[i][0], '\tFinal counts:\t', results[i][1])