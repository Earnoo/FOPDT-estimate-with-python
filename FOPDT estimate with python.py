import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))
def sigmoid_derivative(x, a, b, c):
    return a * c * np.exp(-a * (x - b)) / ((1 + np.exp(-a * (x - b)))**2)

### Read output of step response in matlab *.mat format
mat = scipy.io.loadmat(r'data1.mat')
X1 = mat['t_us']
X2 = mat['y']
df_X0 = pd.DataFrame(X1, columns=['x'])
df_X2 = pd.DataFrame(X2, columns=['y'])

df_X1 = pd.DataFrame()
df_X1['x'] = df_X0['x']
df_X1['y'] = df_X2['y']

plt.scatter(df_X1['x'], df_X1['y'], marker='o', label='X1')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

x_data = df_X1['x']
y_data = df_X1['y']
params, covariance = curve_fit(sigmoid, x_data, y_data, maxfev=10000)
a_opt, b_opt, c_opt = params
derivative_values = sigmoid_derivative(x_data, a_opt, b_opt, c_opt)
y_fit = sigmoid(x_data, a_opt, b_opt, c_opt)

max_slope_index = np.argmax(derivative_values)
max_slope = derivative_values[max_slope_index]
b = y_fit[max_slope_index] - max_slope * x_data[max_slope_index]

x_m = np.linspace(x_data.min(), x_data.max() , 100)
y_m = max_slope * x_m + b

min_slope_index = np.argmin(derivative_values)
min_slope = derivative_values[min_slope_index]
b1 = y_fit[min_slope_index] - min_slope * x_data[min_slope_index]

y_min = min_slope * x_data + b1

y_new = min_slope * x_data + 0.63 * b1


from scipy.optimize import newton
# use 0.63k
target_y = 0.63 * b1 
def diff_function(x):
    return sigmoid(x, a_opt, b_opt, c_opt) - target_y
initial_guess = 9 
k1 = newton(diff_function, initial_guess)

## If want to use 0.28 k
target_y = 0.28 * b1 
def diff_function(x):
    return sigmoid(x, a_opt, b_opt, c_opt) - target_y
initial_guess = 0.0  
k2 = newton(diff_function, initial_guess)


# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, y_fit, label='Fitted Curve', color='red')
plt.plot(x_m, y_m, label='Tangent Line', color='green')
plt.plot(x_data, y_min, label='k', color='blue')
plt.plot(x_data, y_new, label='0.63k', color='black')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()


x_min = 0  # Lower bound
x_max = 20  # Upper bound
x_data_array = x_data.to_numpy() 
# Calculate the area between the two functions
def abs_diff(x):
    return abs(sigmoid(x, a_opt, b_opt, c_opt) - (min_slope * x + b1))


area, _ = quad(abs_diff, x_min, x_max)
print('L = ', -b / max_slope)
print('k = ', b1)
print('Tar base on 0.63k =', k1)
print('Tar base on 2 point(0.63, 0.28) =', 1.5*(k1-k2))
print("Tar base on area:", area / b1 )
print('if we have delay:')
print('T base on 0.63k =', k1- (-b / max_slope))
print('T base on 2 point(0.63, 0.28) =', 1.5*(k1-k2)- (-b / max_slope))
print("T base on area:", (area / b1)- (-b / max_slope))


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))
def sigmoid_derivative(x, a, b, c):
    return a * c * np.exp(-a * (x - b)) / ((1 + np.exp(-a * (x - b)))**2)

## Calculate Errors
X1 = mat['t_us']
X2 = mat['y']

df_X0 = pd.DataFrame(X1, columns=['x'])
df_X2 = pd.DataFrame(X2, columns=['y'])

df_X1 = pd.DataFrame()
df_X1['x'] = df_X0['x']
df_X1['y'] = df_X2['y']
x_data = df_X1['x']



def unit_step(t, a):
    return np.where(t >= a, 1, 0)
def estimate(x):
    return  1.3 * (1 - np.exp(-1.9 * (x - 9))) * unit_step(x, 9)

y_fit = estimate(x_data)




df_X1['est'] = y_fit
df_X1['Error'] = df_X1['y'] - df_X1['est']
mse = np.mean(df_X1['Error'] ** 2)
mae = np.mean(np.abs(df_X1['Error']))
sae = np.sum(np.abs(df_X1['Error']))
sse = np.sum(df_X1['Error'] ** 2)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Sum of Absolute Errors (SAE): {sae}')
print(f'Sum of Squared Errors (SSE): {sse}')