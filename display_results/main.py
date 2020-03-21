import matplotlib.pyplot as plt
import numpy as np


# TODO - Use Google Drive API to connect directly

# Load Control 
control_training = np.load(".\\display_results\\results\\control\\training_accuracy.npy")
control_testing = np.load(".\\display_results\\results\\control\\testing_accuracy.npy")

# Load Parallel
p_control_training = np.load(".\\display_results\\results\\parallel_control\\training_accuracy.npy")
P_control_testing = np.load(".\\display_results\\results\\parallel_control\\testing_accuracy.npy")

# Display Training
x = np.linspace(0,1, len(control_training))
plt.plot(x, control_training)
plt.plot(x, p_control_training)
plt.show()

# Display Testing
x = np.linspace(0,1, len(control_testing))
plt.plot(x, control_testing)
plt.plot(x, P_control_testing)
plt.show()